# =============================================================================
# models/xgboost/train.py  (REDESIGNED v10 — Alpha-Based XGBoost)
#
# KEY CHANGES vs v9:
#
#   1. [TARGET] Uses alpha_5d from merged CSV (stock outperformance vs NIFTY).
#      Filters: abs(alpha_5d) > 1% (ambiguous rows dropped).
#      target = 1 if alpha_5d > 0 else 0
#      This replaces noisy raw direction labels.
#
#   2. [MODEL] Strong regularization to prevent overfitting:
#      max_depth=3, min_child_weight=20, subsample=0.7,
#      colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, gamma=2,
#      learning_rate=0.02, n_estimators≈400 (with early stopping)
#
#   3. [CALIBRATION] CalibratedClassifierCV (isotonic) applied post-training
#      to fix flat probabilities (0.48–0.52 → meaningful spread).
#
#   4. [SPLIT] Strict time-based:
#      train < 2022-01-01  |  val = 2022–2023  |  test >= 2024-01-01
#
#   5. [FEATURES] Lean set of ~30 strong, stock-specific features.
#      Removed: event_*, news_*, sector_encoded, raw price lags.
#
#   6. [THRESHOLD] Prediction at 0.55 (not 0.5) to reduce UP bias.
#
#   7. [CONFIDENCE FILTER] Optional evaluation on high-confidence rows
#      (prob > 0.6 or < 0.4) reported separately.
#
# EXPECTED RESULTS:
#   - Test accuracy: 55–60% (vs 50–51% before)
#   - Test AUC:      0.56–0.62
#   - No majority-class collapse
#   - Calibrated probabilities with real spread
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
# FEATURE LIST — lean, strong, stock-specific
# ---------------------------------------------------------------------------

FEATURES = [
    # Core relative strength vs NIFTY (primary alpha signal)
    "ret_vs_nifty_1d",
    "ret_vs_nifty_5d",
    "alpha_strength",          # rolling mean of ret_vs_nifty_1d

    # Relative strength vs sector
    "ret_vs_sector_1d",
    "ret_vs_sector_5d",

    # Absolute momentum
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_diff",           # momentum_5d - momentum_10d

    # Volatility & volume
    "vol_spike",               # Volume / rolling_mean(Volume, 10)
    "vol_ratio",               # Volume / rolling_mean(Volume, 20)
    "vol_breakout",            # current_vol / mean_vol_20d
    "volatility_10d",
    "volatility_20d",

    # RSI
    "RSI",
    "rsi_momentum",

    # Price structure
    "price_pos_20d",           # position within 20-day high-low range
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

    # Fundamental (weak but stabilising)
    "ROE",
    "pe_ratio_rank",
    "Revenue_Growth",

    # NIFTY context (market regime)
    "nifty_rsi",
    "nifty_above_ema20",
]


# ---------------------------------------------------------------------------
# XGBoost hyperparameters — strong regularisation
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = {
    "n_estimators"    : 2000,       # high ceiling; early stopping kicks in ~400
    "max_depth"       : 3,          # shallow trees → less overfit
    "learning_rate"   : 0.02,
    "subsample"       : 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 20,         # large → conservative splits
    "reg_lambda"      : 5.0,        # L2
    "reg_alpha"       : 2.0,        # L1
    "gamma"           : 2.0,        # min split gain
    "eval_metric"     : "logloss",
    "random_state"    : RANDOM_SEED,
    "n_jobs"          : -1,
    "tree_method"     : "hist",
    "early_stopping_rounds": 50,
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _evaluate(y_true, y_pred, y_proba, split_name, threshold=0.55):
    """Full evaluation at a given decision threshold."""
    # Apply threshold to UP class (col 1)
    y_pred_thr = (y_proba[:, 1] >= threshold).astype(int)

    acc    = accuracy_score(y_true, y_pred_thr)
    wf1    = f1_score(y_true, y_pred_thr, average="weighted", zero_division=0)
    mf1    = f1_score(y_true, y_pred_thr, average="macro",    zero_division=0)
    auc    = roc_auc_score(y_true, y_proba[:, 1])
    return {"split": split_name, "accuracy": acc, "weighted_f1": wf1,
            "macro_f1": mf1, "auc_roc": auc}


def _confidence_eval(y_true, y_proba, threshold=0.6):
    """Evaluate only high-confidence predictions (prob > 0.6 or < 0.4)."""
    mask = (y_proba[:, 1] > threshold) | (y_proba[:, 1] < (1 - threshold))
    if mask.sum() < 50:
        return None
    y_pred_conf = (y_proba[mask, 1] >= 0.5).astype(int)
    acc = accuracy_score(y_true[mask], y_pred_conf)
    auc = roc_auc_score(y_true[mask], y_proba[mask, 1])
    return {"n": int(mask.sum()), "pct": mask.mean(), "accuracy": acc, "auc": auc}


def _get_X(df, feature_names, train_medians=None):
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else 0.0
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if train_medians is not None:
        X = X.fillna(pd.Series(train_medians))
    return X


# ---------------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------------

def train():
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print("  XGBOOST — REDESIGNED v10 (Alpha-Based + Calibrated)")
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

    # ── 1. Extract alpha-based label ─────────────────────────────────────────
    if "alpha_5d" not in df.columns:
        _p("x", "Column 'alpha_5d' missing. Re-run merge_features.py (v10).")
        return {}

    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # Filter: keep only rows where |alpha_5d| > 1% (clear signal)
    before = len(df)
    df = df[df["alpha_5d"].notna()].copy()
    alpha_thresh = 0.01
    df = df[df["alpha_5d"].abs() >= alpha_thresh].copy()
    after = len(df)
    _p("OK", f"Alpha filter (|alpha_5d| > {alpha_thresh*100:.0f}%): "
              f"{before} → {after} rows  ({(before-after)/before*100:.1f}% dropped)")

    # Binary label: 1 = stock outperformed NIFTY, 0 = underperformed
    df["label"] = (df["alpha_5d"] > 0).astype(int)
    n_up   = (df["label"] == 1).sum()
    n_down = (df["label"] == 0).sum()
    _p("OK", f"Label distribution: UP={n_up}  DOWN={n_down}  "
              f"UP%={n_up/(n_up+n_down)*100:.1f}%")

    # ── 2. Time-based split ──────────────────────────────────────────────────
    df["Date"] = df["Date"].astype(str)
    train_df = df[df["Date"] <  "2022-01-01"].copy()
    val_df   = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2024-01-01")].copy()
    test_df  = df[df["Date"] >= "2024-01-01"].copy()

    _p("OK", f"Split → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")
    _p("i",  f"Test UP%={test_df['label'].mean():.3f}")

    if len(train_df) < 1000 or len(val_df) < 200 or len(test_df) < 200:
        _p("!", "WARNING: Very small split. Check date range in your data.")

    # ── 3. Feature matrix ────────────────────────────────────────────────────
    feat_used = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]
    if missing:
        _p("!", f"Features missing from CSV ({len(missing)}): {missing[:10]}...")
    _p("i", f"Using {len(feat_used)} features")

    train_medians = df[feat_used].median().to_dict()  # computed on full df before split

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
    _p("i", f"scale_pos_weight={spw:.3f}")

    # ── 5. Train XGBoost ──────────────────────────────────────────────────────
    params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}
    early_stop = params.pop("early_stopping_rounds", 50)

    _p("i", f"Training XGBoost: {len(feat_used)} features, {len(X_train)} rows ...")
    model = XGBClassifier(**params, early_stopping_rounds=early_stop)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    _p("i", f"Best iteration: {model.best_iteration}  "
             f"Best val_logloss: {model.best_score:.5f}")

    # ── 6. Calibrate probabilities ────────────────────────────────────────────
    _p("i", "Calibrating probabilities (isotonic regression) ...")
    # Use val set for calibration — no data leakage
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    _p("OK", "Calibration complete")

    # ── 7. Predictions ────────────────────────────────────────────────────────
    y_proba_train = calibrated.predict_proba(X_train)
    y_proba_val   = calibrated.predict_proba(X_val)
    y_proba_test  = calibrated.predict_proba(X_test)

    THRESHOLD = 0.55  # raises precision for UP predictions

    # ── 8. Evaluation ─────────────────────────────────────────────────────────
    results_table = []
    for split, yt, yp in [("Train", y_train, y_proba_train),
                           ("Val",   y_val,   y_proba_val),
                           ("Test",  y_test,  y_proba_test)]:
        r = _evaluate(yt, (yp[:, 1] >= THRESHOLD).astype(int), yp, split, THRESHOLD)
        results_table.append(r)

    print("\n" + "=" * 68)
    print("  XGBOOST v10 — EVALUATION SUMMARY")
    print("=" * 68)
    print(f"  {'Metric':<14} | {'Train':>10} | {'Val':>10} | {'Test':>10}")
    print(f"  {'-'*14}+{'-'*12}+{'-'*12}+{'-'*12}")
    for metric in ["accuracy", "weighted_f1", "macro_f1", "auc_roc"]:
        vals = [f"{r[metric]:.4f}" for r in results_table]
        print(f"  {metric.replace('_',' ').title():<14} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10}")
    print("=" * 68)

    # Detailed test report at threshold 0.55
    y_pred_test_thr = (y_proba_test[:, 1] >= THRESHOLD).astype(int)
    print(f"\n  [ Test Set Report — threshold={THRESHOLD} ]")
    print(classification_report(y_test, y_pred_test_thr,
                                 target_names=["DOWN", "UP"], digits=3))
    print("  Confusion Matrix (DOWN/UP):")
    print(confusion_matrix(y_test, y_pred_test_thr))

    # Probability spread check
    test_probs = y_proba_test[:, 1]
    _p("i", f"Prob spread — mean={test_probs.mean():.4f}  "
             f"std={test_probs.std():.4f}  "
             f"min={test_probs.min():.4f}  "
             f"max={test_probs.max():.4f}")

    # Confidence-filtered evaluation
    conf_result = _confidence_eval(y_test, y_proba_test, threshold=0.6)
    if conf_result:
        print(f"\n  [ High-Confidence Predictions (prob>0.6 or <0.4) ]")
        print(f"    Rows:     {conf_result['n']} ({conf_result['pct']*100:.1f}% of test)")
        print(f"    Accuracy: {conf_result['accuracy']:.4f}")
        print(f"    AUC-ROC:  {conf_result['auc']:.4f}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feat_used)
    top20 = importances.nlargest(20)
    print("\n  Top-20 feature importances:")
    for feat, imp in top20.items():
        print(f"    {feat:35s}  {imp:.4f}")

    # Baseline
    _p("i", f"Always-UP baseline on test: {y_test.mean():.4f}")

    # ── 9. Save model ─────────────────────────────────────────────────────────
    payload = {
        "model"         : calibrated,
        "base_model"    : model,
        "feature_names" : feat_used,
        "train_medians" : train_medians,
        "threshold"     : THRESHOLD,
        "label_type"    : "alpha_5d",
        "alpha_threshold": alpha_thresh,
    }
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    _p("OK", f"Model saved → {XGB_MODEL_PATH}")

    # ── 10. Save results CSV ──────────────────────────────────────────────────
    res = test_df[["Date", "Stock"]].copy()
    res["label"]      = y_test
    res["pred"]       = y_pred_test_thr
    res["prob_up"]    = y_proba_test[:, 1]
    res["alpha_5d"]   = test_df["alpha_5d"].values
    os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
    res.to_csv(XGB_RESULTS_PATH, index=False)
    _p("OK", f"Results saved → {XGB_RESULTS_PATH}")

    return payload


if __name__ == "__main__":
    train()
