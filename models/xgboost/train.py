# =============================================================================
# models/xgboost/train.py  (FIXED v13)
#
# FIXES vs v12:
#   FIX 1: REMOVED CalibratedClassifierCV (crashed with cv="prefit")
#   FIX 2: Stronger XGBClassifier params — deeper trees, less regularization
#   FIX 3: Alpha threshold raised to 0.015 (matches merge_features v13)
#   FIX 4: FEATURES pruned to strong set (imported from merge_features)
#   FIX 5: Output raw XGBoost probabilities — no calibration wrapper
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report, confusion_matrix)

from config.settings import (MERGED_CSV, XGB_MODEL_PATH, XGB_RESULTS_PATH,
                              RANDOM_SEED)


def _p(tag, msg):
    print(f"  [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# FEATURE LIST (strong set — matches merge_features v13 FEATURES)
# ---------------------------------------------------------------------------

FEATURES = [
    "RSI", "MACD_hist", "EMA_20", "ATR", "OBV",
    "news_score", "news_rolling_3d",
    "event_score_max", "is_event",
    "ret_vs_nifty_1d", "ret_vs_nifty_5d",
    "ret_vs_sector_1d", "ret_vs_sector_5d",
    "alpha_strength",
    "vol_spike", "vol_breakout",
    "momentum_diff", "price_pos_20d",
    "PE_Ratio", "ROE",
    "momentum_strength",
    "volatility_ratio",
]

# FIX 3: Stronger alpha threshold
ALPHA_THRESH = 0.015
THRESHOLD    = 0.52

# Splits
TRAIN_END  = "2023-12-31"
VAL_START  = "2024-01-01"
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"


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
    acc  = accuracy_score(y_true, y_pred)
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mf1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    auc  = roc_auc_score(y_true, y_proba[:, 1])
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
    print("  XGBOOST — FIXED v13 (no calibration, strong params)")
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
        _p("!", f"WARNING: {pre_2020} rows before 2020-01-01. Re-run merge_features.py v13.")

    # ── 1. Alpha-based label (FIX 3: threshold=0.015) ──────────────────────
    if "alpha_5d" not in df.columns:
        _p("x", "Column 'alpha_5d' missing. Re-run merge_features.py v13.")
        return {}

    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    before = len(df)
    df = df[df["alpha_5d"].notna()].copy()
    df = df[df["alpha_5d"].abs() >= ALPHA_THRESH].copy()
    after = len(df)
    _p("OK", f"Alpha filter (|alpha_5d| > {ALPHA_THRESH*100:.1f}%): "
              f"{before} → {after} rows  ({(before-after)/before*100:.1f}% dropped)")

    df["label"] = (df["alpha_5d"] > 0).astype(int)
    n_up   = (df["label"] == 1).sum()
    n_down = (df["label"] == 0).sum()
    _p("OK", f"Label distribution: UP={n_up}  DOWN={n_down}  "
              f"UP%={n_up/(n_up+n_down)*100:.1f}%")

    # ── 2. Time-based split ─────────────────────────────────────────────────
    df["Date"] = df["Date"].astype(str)
    train_df = df[df["Date"] <= TRAIN_END].copy()
    val_df   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test_df  = df[df["Date"] >= TEST_START].copy()

    _p("OK", f"Train: {len(train_df)} rows  ({train_df['Date'].min()} → {TRAIN_END})")
    _p("OK", f"Val:   {len(val_df)} rows   ({VAL_START} → {VAL_END})")
    _p("OK", f"Test:  {len(test_df)} rows  ({TEST_START} →)")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(split_df) < 200:
            _p("!", f"WARNING: {split_name} set has only {len(split_df)} rows after alpha filter.")

    # ── 3. Feature matrix ───────────────────────────────────────────────────
    feat_used = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        _p("!", f"Features missing from CSV ({len(missing)}): {missing}")
    _p("i", f"Using {len(feat_used)} features")

    # Train-only medians (no leakage)
    train_medians = train_df[feat_used].apply(pd.to_numeric, errors="coerce").median().to_dict()

    X_train = _get_X(train_df, feat_used, train_medians)
    X_val   = _get_X(val_df,   feat_used, train_medians)
    X_test  = _get_X(test_df,  feat_used, train_medians)

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ── 4. Class balance ─────────────────────────────────────────────────────
    n_tr_up   = (y_train == 1).sum()
    n_tr_down = (y_train == 0).sum()
    spw = n_tr_down / max(n_tr_up, 1)
    _p("i", f"scale_pos_weight={spw:.3f} (train: UP={n_tr_up}, DOWN={n_tr_down})")

    # ── 5. Train XGBoost (FIX 1+2: no calibration, stronger params) ─────────
    # FIX 2: Params from spec — deeper trees, lower regularization
    model = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        eval_metric="logloss",
        scale_pos_weight=spw,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    _p("i", f"Training XGBoost: {len(feat_used)} features, {len(X_train)} rows ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50,
    )
    _p("i", f"Best iteration: {model.best_iteration}  "
             f"Best val_logloss: {model.best_score:.5f}")

    if model.best_iteration == 0:
        _p("!", "WARNING: best_iteration=0 — model did not improve. Check features/data.")

    # ── 6. Predictions (FIX 1+5: raw probabilities, NO calibration wrapper) ──
    y_proba_train = model.predict_proba(X_train)
    y_proba_val   = model.predict_proba(X_val)
    y_proba_test  = model.predict_proba(X_test)

    # ── 7. Evaluation ─────────────────────────────────────────────────────────
    results_list = []
    for split, yt, yp in [("Train", y_train, y_proba_train),
                            ("Val",   y_val,   y_proba_val),
                            ("Test",  y_test,  y_proba_test)]:
        r = _evaluate(yt, yp, split)
        results_list.append(r)

    print("\n" + "=" * 68)
    print("  XGBOOST v13 — EVALUATION SUMMARY")
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

    # ── 8. Save (FIX 5: save raw model, not calibrated wrapper) ──────────────
    payload = {
        "model"          : model,           # raw XGBClassifier
        "feature_names"  : feat_used,
        "train_medians"  : train_medians,
        "threshold"      : THRESHOLD,
        "label_type"     : "alpha_5d",
        "alpha_threshold": ALPHA_THRESH,
        "split": {
            "train_end" : TRAIN_END,
            "val_start" : VAL_START,
            "val_end"   : VAL_END,
            "test_start": TEST_START,
        },
        # expose val probabilities for ensemble stacking
        "val_proba": y_proba_val,
        "val_labels": y_val,
    }
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    _p("OK", f"Model saved → {XGB_MODEL_PATH}")

    # ── 9. Save results ────────────────────────────────────────────────────────
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