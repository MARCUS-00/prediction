# =============================================================================
# models/xgboost/train.py  (FIXED v12)
#
# FIXES vs v11:
#
#   FIX 1: Time-based split matches project requirement exactly.
#     train: 2020-01-01 to 2023-12-31
#     val:   2024-01-01 to 2024-12-31
#     test:  2025-01-01 to 2025-12-31
#
#   FIX 2: news_score_daily and news_rolling_3d are now in merged_final.csv
#     (fixed in merge_features.py v12). They are included as features.
#
#   FIX 3: event_score_max and is_event now correctly merged (fix in
#     merge_features.py v12). They are included as features.
#
#   All v11 fixes retained:
#     - train_medians computed from TRAIN SPLIT ONLY (no leakage)
#     - CalibratedClassifierCV with isotonic method
#     - scale_pos_weight computed dynamically
#     - Alpha-based binary target (stock outperforms NIFTY by >1%)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)

from config.settings import (MERGED_CSV, XGB_MODEL_PATH, XGB_RESULTS_PATH,
                              RANDOM_SEED)


def _p(tag, msg):
    print(f"  [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# FEATURE LIST
# ---------------------------------------------------------------------------

FEATURES = [
    # Core relative strength vs NIFTY (primary alpha signal)
    "ret_vs_nifty_1d",
    "ret_vs_nifty_5d",
    "alpha_strength",

    # Relative strength vs sector
    "ret_vs_sector_1d",
    "ret_vs_sector_5d",

    # Absolute momentum
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_diff",

    # Volatility & volume
    "vol_spike",
    "vol_ratio",
    "vol_breakout",
    "volatility_10d",
    "volatility_20d",

    # RSI
    "RSI",
    "rsi_momentum",

    # Price structure
    "price_pos_20d",
    "pct_from_52w_high",
    "pct_from_52w_low",
    "close_range_pct",
    "hl_ratio",
    "gap_pct",
    "atr_ratio",

    # EMA trend
    "ema_dist_20",
    "ema_dist_50",
    "ema_cross",

    # Sharpe
    "sharpe_5d",

    # Cross-sectional ranks
    "cs_rank_momentum_5d",
    "cs_rank_vol_spike",
    "cs_rank_RSI",
    "cs_rank_ret_vs_nifty_5d",

    # Fundamental
    "ROE",
    "pe_ratio_rank",
    "Revenue_Growth",

    # NIFTY market context
    "nifty_rsi",
    "nifty_above_ema20",

    # News (FIX 2: now present in merged_final.csv)
    "news_score_daily",
    "news_rolling_3d",
    "news_spike",
    "has_news",

    # Events (FIX 3: now correctly merged)
    "event_score_max",
    "is_event",
]

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = {
    "n_estimators"    : 2000,
    "max_depth"       : 3,
    "learning_rate"   : 0.02,
    "subsample"       : 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,
    "reg_lambda"      : 5.0,
    "reg_alpha"       : 2.0,
    "gamma"           : 2.0,
    "eval_metric"     : "logloss",
    "random_state"    : RANDOM_SEED,
    "n_jobs"          : -1,
    "tree_method"     : "hist",
    "early_stopping_rounds": 50,
}

# FIX 1: Correct split for 2020-2025 data
TRAIN_END  = "2023-12-31"   # train: 2020-2023 (4 years)
VAL_START  = "2024-01-01"   # val:   2024
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"   # test:  2025

ALPHA_THRESH = 0.01      # keep only |alpha_5d| > 1%
THRESHOLD    = 0.55      # decision threshold for UP class


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_X(df, feature_names, train_medians):
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = train_medians.get(col, 0.0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(pd.Series(train_medians))
    return X


def _evaluate(y_true, y_proba, split_name):
    y_pred = (y_proba[:, 1] >= THRESHOLD).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    wf1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mf1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    auc    = roc_auc_score(y_true, y_proba[:, 1])
    return {"split": split_name, "accuracy": acc,
            "weighted_f1": wf1, "macro_f1": mf1, "auc_roc": auc}


def _confidence_eval(y_true, y_proba, threshold=0.6):
    mask = (y_proba[:, 1] > threshold) | (y_proba[:, 1] < (1 - threshold))
    if mask.sum() < 50:
        return None
    y_pred = (y_proba[mask, 1] >= 0.5).astype(int)
    return {
        "n":        int(mask.sum()),
        "pct":      float(mask.mean()),
        "accuracy": accuracy_score(y_true[mask], y_pred),
        "auc":      roc_auc_score(y_true[mask], y_proba[mask, 1]),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def train():
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print("  XGBOOST — FIXED v12 (2020-2025 split, news+events fixed)")
    print("=" * 60)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        _p("x", "xgboost not installed. Run: pip install xgboost")
        return {}

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found. Run merge_features.py first.")
        return {}

    df = pd.read_csv(MERGED_CSV)
    _p("OK", f"Loaded merged_final.csv  shape={df.shape}")

    dates = pd.to_datetime(df["Date"])
    _p("i", f"Date range: {dates.min().date()} → {dates.max().date()}")
    pre_2020 = (dates < "2020-01-01").sum()
    if pre_2020 > 0:
        _p("!", f"WARNING: {pre_2020} rows before 2020-01-01. Re-run merge_features.py v12.")

    # Validate news features present
    for feat in ["news_score_daily", "news_rolling_3d"]:
        if feat not in df.columns:
            _p("!", f"WARNING: {feat} missing from merged CSV. Re-run merge_features.py v12.")

    # ── 1. Alpha-based label ─────────────────────────────────────────────────
    if "alpha_5d" not in df.columns:
        _p("x", "Column 'alpha_5d' missing. Re-run merge_features.py v12.")
        return {}

    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    before = len(df)
    df = df[df["alpha_5d"].notna()].copy()
    df = df[df["alpha_5d"].abs() >= ALPHA_THRESH].copy()
    after = len(df)
    _p("OK", f"Alpha filter (|alpha_5d| > {ALPHA_THRESH*100:.0f}%): "
              f"{before} → {after} rows  ({(before-after)/before*100:.1f}% dropped)")

    df["label"] = (df["alpha_5d"] > 0).astype(int)
    n_up   = (df["label"] == 1).sum()
    n_down = (df["label"] == 0).sum()
    _p("OK", f"Label distribution: UP={n_up}  DOWN={n_down}  "
              f"UP%={n_up/(n_up+n_down)*100:.1f}%")

    # ── 2. Time-based split (FIX 1) ─────────────────────────────────────────
    df["Date"] = df["Date"].astype(str)
    train_df = df[df["Date"] <= TRAIN_END].copy()
    val_df   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test_df  = df[df["Date"] >= TEST_START].copy()

    _p("OK", "Split (FIX 1: matches project requirement):")
    _p("OK", f"  Train: {len(train_df)} rows ({train_df['Date'].min()} → {train_df['Date'].max()})")
    _p("OK", f"  Val:   {len(val_df)} rows ({val_df['Date'].min()} → {val_df['Date'].max()})")
    _p("OK", f"  Test:  {len(test_df)} rows ({test_df['Date'].min()} → {test_df['Date'].max()})")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(split_df) < 200:
            _p("!", f"WARNING: {split_name} set has only {len(split_df)} rows after alpha filter.")

    _p("i", f"Test UP%={test_df['label'].mean():.3f}  "
             f"Val UP%={val_df['label'].mean():.3f}")

    # ── 3. Feature matrix ─────────────────────────────────────────────────────
    feat_used = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        _p("!", f"Features missing from CSV ({len(missing)}): {missing}")
    _p("i", f"Using {len(feat_used)} features")

    # train_medians from TRAIN SPLIT ONLY (no leakage)
    train_medians = train_df[feat_used].apply(pd.to_numeric, errors="coerce").median().to_dict()
    _p("OK", "train_medians computed from TRAIN SPLIT ONLY (no leakage)")

    X_train = _get_X(train_df, feat_used, train_medians)
    X_val   = _get_X(val_df,   feat_used, train_medians)
    X_test  = _get_X(test_df,  feat_used, train_medians)

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ── 4. Class balance ──────────────────────────────────────────────────────
    n_tr_up   = (y_train == 1).sum()
    n_tr_down = (y_train == 0).sum()
    spw = n_tr_down / max(n_tr_up, 1)
    _p("i", f"scale_pos_weight={spw:.3f} (train: UP={n_tr_up}, DOWN={n_tr_down})")

    # ── 5. Train XGBoost ──────────────────────────────────────────────────────
    params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}
    early_stop = params.pop("early_stopping_rounds", 50)

    _p("i", f"Training XGBoost: {len(feat_used)} features, {len(X_train)} rows ...")
    model = XGBClassifier(**params, early_stopping_rounds=early_stop)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    _p("i", f"Best iteration: {model.best_iteration}  "
             f"Best val_logloss: {model.best_score:.5f}")

    # ── 6. Calibrate ──────────────────────────────────────────────────────────
    _p("i", "Calibrating probabilities (isotonic regression on val set) ...")
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    _p("OK", "Calibration complete")

    # ── 7. Predictions ────────────────────────────────────────────────────────
    y_proba_train = calibrated.predict_proba(X_train)
    y_proba_val   = calibrated.predict_proba(X_val)
    y_proba_test  = calibrated.predict_proba(X_test)

    # ── 8. Evaluation ─────────────────────────────────────────────────────────
    results_list = []
    for split, yt, yp in [("Train", y_train, y_proba_train),
                            ("Val",   y_val,   y_proba_val),
                            ("Test",  y_test,  y_proba_test)]:
        r = _evaluate(yt, yp, split)
        results_list.append(r)

    print("\n" + "=" * 68)
    print("  XGBOOST v12 — EVALUATION SUMMARY")
    print("=" * 68)
    print(f"  {'Metric':<14} | {'Train':>10} | {'Val':>10} | {'Test':>10}")
    print(f"  {'-'*14}+{'-'*12}+{'-'*12}+{'-'*12}")
    for metric in ["accuracy", "weighted_f1", "macro_f1", "auc_roc"]:
        vals = [f"{r[metric]:.4f}" for r in results_list]
        print(f"  {metric.replace('_',' ').title():<14} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10}")
    print("=" * 68)

    y_pred_test = (y_proba_test[:, 1] >= THRESHOLD).astype(int)
    print(f"\n  [ Test Set Report — threshold={THRESHOLD} ]")
    print(classification_report(y_test, y_pred_test,
                                 target_names=["DOWN", "UP"], digits=3))
    print("  Confusion Matrix (rows=actual DOWN/UP, cols=predicted DOWN/UP):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"    {cm}")

    test_probs = y_proba_test[:, 1]
    _p("i", f"Prob spread — mean={test_probs.mean():.4f}  "
             f"std={test_probs.std():.4f}  "
             f"min={test_probs.min():.4f}  "
             f"max={test_probs.max():.4f}")
    if test_probs.std() < 0.02:
        _p("!", "WARNING: std < 0.02 — model may be collapsing to one class.")

    conf = _confidence_eval(y_test, y_proba_test, threshold=0.6)
    if conf:
        print(f"\n  [ High-Confidence Predictions (prob > 0.6 or < 0.4) ]")
        print(f"    Rows:     {conf['n']} ({conf['pct']*100:.1f}% of test)")
        print(f"    Accuracy: {conf['accuracy']:.4f}")
        print(f"    AUC-ROC:  {conf['auc']:.4f}")

    baseline_acc = float(y_test.mean())
    _p("i", f"Always-UP baseline on test: {baseline_acc:.4f}")
    _p("i", f"Model improvement over baseline: {results_list[2]['accuracy'] - baseline_acc:+.4f}")

    importances = pd.Series(model.feature_importances_, index=feat_used)
    top20 = importances.nlargest(20)
    print("\n  Top-20 feature importances:")
    for feat, imp in top20.items():
        print(f"    {feat:35s}  {imp:.4f}")

    # ── 9. Save ────────────────────────────────────────────────────────────────
    payload = {
        "model"          : calibrated,
        "base_model"     : model,
        "feature_names"  : feat_used,
        "train_medians"  : train_medians,
        "threshold"      : THRESHOLD,
        "label_type"     : "alpha_5d",
        "alpha_threshold": ALPHA_THRESH,
        "split"          : {
            "train_end" : TRAIN_END,
            "val_start" : VAL_START,
            "val_end"   : VAL_END,
            "test_start": TEST_START,
        },
    }
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    _p("OK", f"Model saved → {XGB_MODEL_PATH}")

    # ── 10. Save results ───────────────────────────────────────────────────────
    res = test_df[["Date", "Stock"]].copy()
    res["label"]    = y_test
    res["pred"]     = y_pred_test
    res["prob_up"]  = y_proba_test[:, 1]
    res["alpha_5d"] = test_df["alpha_5d"].values
    os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
    res.to_csv(XGB_RESULTS_PATH, index=False)
    _p("OK", f"Results saved → {XGB_RESULTS_PATH}")

    return payload


if __name__ == "__main__":
    train()