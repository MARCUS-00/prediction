# =============================================================================
# models/ensemble/train_meta.py  (FIXED v12)
#
# FIXES vs v9:
#
#   FIX 1: Date-based split consistent with XGBoost and LSTM:
#     train: 2020-2023, val: 2024, test: 2025
#     Old code used ratio-based split — inconsistent with base models.
#
#   FIX 2: Meta-learner trained on VAL set predictions from base models.
#     This is the correct stacking protocol: base models see train only;
#     meta-learner sees val (out-of-fold) predictions; test is truly unseen.
#
#   FIX 3: news_proba uses news_score (aliased column, always present).
#
#   All v9 fixes retained:
#     - 5-day label with 0.5% threshold
#     - LogisticRegression meta-learner with calibration
#     - Fallback weighted average if meta-train too small
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from config.settings         import (MERGED_CSV, LABEL_THRESHOLD, LABEL_HORIZON,
                                      META_MODEL_PATH, ENSEMBLE_RESULTS_PATH,
                                      RANDOM_SEED)
from models.xgboost.predict  import predict_proba as xgb_proba, load_xgb
from models.lstm.predict     import predict_proba  as lstm_proba, load_lstm
from evaluation.metrics      import evaluate_all


def _p(tag, msg): print(f"  [{tag}] {msg}", flush=True)

# FIX 1: Date-based split (matches XGBoost and LSTM)
TRAIN_END  = "2023-12-31"
VAL_START  = "2024-01-01"
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"


def _apply_threshold_labels(df):
    """5-day forward label with 0.5% threshold. Consistent with XGB/LSTM."""
    df = df.copy().sort_values(["Stock", "Date"]).reset_index(drop=True)
    df["_close_future"] = df.groupby("Stock")["Close"].shift(-LABEL_HORIZON)
    df["_return_pct"]   = (df["_close_future"] - df["Close"]) / df["Close"].replace(0, np.nan)
    mask = df["_return_pct"].abs() >= LABEL_THRESHOLD
    df   = df[mask].copy()
    df["label"] = np.where(df["_return_pct"] > 0, 1, 0).astype(int)
    df.drop(columns=["_close_future", "_return_pct"], inplace=True)
    return df


def _date_split(df):
    """FIX 1: strict date-based split."""
    train_df = df[df["Date"] <= TRAIN_END].copy()
    val_df   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test_df  = df[df["Date"] >= TEST_START].copy()
    return train_df, val_df, test_df


def _news_proba(df):
    """
    FIX 3: Use 'news_score' column (aliased from news_score_daily in merge v12).
    Always present in merged_final.csv after merge_features.py v12.
    """
    score_col = None
    if "news_score" in df.columns:
        score_col = "news_score"
    elif "news_score_daily" in df.columns:
        score_col = "news_score_daily"

    if score_col:
        score = df[score_col].fillna(0.0).values.astype(np.float32)
        p_up  = 1.0 / (1.0 + np.exp(-score * 3))   # sigmoid scaling
        return np.column_stack([1.0 - p_up, p_up])
    return np.full((len(df), 2), 0.5)


def train():
    print("\n" + "=" * 55)
    print("  ENSEMBLE META-LEARNER — Training (FIXED v12)")
    print("=" * 55)

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("OK", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df = df.sort_values("Date").reset_index(drop=True)

    # FIX 2: 5-day label with 0.5% threshold
    before = len(df)
    df = _apply_threshold_labels(df)
    _p("OK", f"Threshold filter: {before} → {len(df)} rows")

    # FIX 1: date-based split
    _, val_df, test_df = _date_split(df)

    # Meta-learner trains on VAL set predictions (correct stacking protocol)
    meta_train_df = val_df.copy()
    _p("OK", f"Meta-train (val predictions): {len(meta_train_df)}  Test: {len(test_df)}")
    _p("i", f"Date split: val={VAL_START}→{VAL_END}  test={TEST_START}→end")

    try:    xgb  = load_xgb()
    except Exception as e: _p("!", f"XGBoost load failed: {e}"); xgb = None

    try:    lstm = load_lstm()
    except Exception as e: _p("!", f"LSTM load failed: {e}"); lstm = None

    def _get_parts(subset_df):
        parts = []
        for name, fn, payload in [("XGBoost", xgb_proba, xgb), ("LSTM", lstm_proba, lstm)]:
            if payload:
                try:
                    p = fn(subset_df, payload)
                    parts.append(p)
                    _p("OK", f"{name} probabilities: {p.shape}")
                except Exception as e:
                    _p("!", f"{name} predict failed: {e}; uniform fill")
                    parts.append(np.full((len(subset_df), 2), 0.5))
            else:
                _p("!", f"{name} not loaded; uniform fill")
                parts.append(np.full((len(subset_df), 2), 0.5))
        parts.append(_news_proba(subset_df))
        # shape: (n, 6) = [xgb_down, xgb_up, lstm_down, lstm_up, news_down, news_up]
        return np.hstack(parts)

    X_meta = _get_parts(meta_train_df)
    X_test = _get_parts(test_df)
    y_meta = meta_train_df["label"].values
    y_test = test_df["label"].values

    _p("i", f"Meta-feature matrix: train={X_meta.shape}  test={X_test.shape}")

    _p("i", "Training Logistic Regression meta-learner ...")
    base_lr = LogisticRegression(
        max_iter=2000, random_state=RANDOM_SEED, C=0.1,
        solver="lbfgs", class_weight="balanced",
    )

    class_counts = np.bincount(y_meta, minlength=2)

    if len(meta_train_df) < 50:
        _p("!", "Meta-train too small; using weighted average ensemble")

        def _weighted_avg(X_feat):
            p_up = 0.50 * X_feat[:, 1] + 0.35 * X_feat[:, 3] + 0.15 * X_feat[:, 5]
            return np.column_stack([1 - np.clip(p_up, 0, 1), np.clip(p_up, 0, 1)])

        class _WeightedAvgModel:
            def predict_proba(self, X): return _weighted_avg(X)
            def predict(self, X): return (_weighted_avg(X)[:, 1] >= 0.5).astype(int)
            def fit(self, X, y): return self

        meta = _WeightedAvgModel()
    elif class_counts.min() >= 3:
        meta = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")
    else:
        meta = base_lr

    meta.fit(X_meta, y_meta)

    y_pred_meta  = meta.predict(X_meta)
    y_proba_meta = meta.predict_proba(X_meta)
    y_pred_test  = meta.predict(X_test)
    y_proba_test = meta.predict_proba(X_test)

    metrics = evaluate_all(
        y_meta, y_pred_meta, y_proba_meta,
        y_meta, y_pred_meta, y_proba_meta,
        y_test, y_pred_test, y_proba_test,
        "Ensemble"
    )

    payload = {"meta_model": meta, "metrics": metrics}
    try:
        os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
        with open(META_MODEL_PATH, "wb") as f: pickle.dump(payload, f)
        _p("OK", f"Model -> {META_MODEL_PATH}")
    except Exception as e: _p("x", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date", "Stock"]].copy()
        res["label"]      = y_test
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res["prob_up"]    = y_proba_test[:, 1]
        res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
        _p("OK", f"Results -> {ENSEMBLE_RESULTS_PATH}")
    except Exception as e: _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()