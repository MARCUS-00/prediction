# =============================================================================
# models/ensemble/train_meta.py  (FIXED v9)
#
# Critical fixes vs v8:
#
#   FIX 1: LSTM CRASH — "_build_sequences() missing 1 required positional
#     argument: 'seq_len'" — fixed in lstm/train.py (seq_len now has default)
#     AND lstm/predict.py now passes seq_len from loaded payload. This was
#     causing LSTM to always fall back to uniform 0.5 probabilities.
#
#   FIX 2: LABEL MISMATCH — ensemble was using 1-day forward label
#     (_close_next = shift(-1)) while XGBoost and LSTM use 5-day.
#     Fixed to match: shift(-LABEL_HORIZON) with LABEL_THRESHOLD=0.005.
#
#   FIX 3: META-LEARNER — increased meta-train to full val set (was 70%).
#     With the LSTM fix, the meta-learner now sees 3 real signal columns
#     (XGB, LSTM, news) instead of [XGB, 0.5, news]. This makes the
#     logistic regression weights meaningful.
#
#   FIX 4: feature engineering — uses the same _engineer_features from
#     xgboost/train.py which now also computes return_vs_sector and
#     news_rolling_3d so base models receive their expected features.
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
                                      TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
from models.xgboost.predict  import predict_proba as xgb_proba, load_xgb
from models.lstm.predict     import predict_proba  as lstm_proba, load_lstm
from models.xgboost.train    import _engineer_features   # shared feature pipeline
from evaluation.metrics      import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}", flush=True)


def _apply_threshold_labels(df):
    """
    FIX 2: Use 5-day forward label with 0.5% threshold.
    Was using 1-day forward (shift(-1)) — mismatch with base models.
    """
    df = df.copy().sort_values(["Stock", "Date"]).reset_index(drop=True)
    df["_close_future"] = df.groupby("Stock")["Close"].shift(-LABEL_HORIZON)
    df["_return_pct"]   = (df["_close_future"] - df["Close"]) / df["Close"].replace(0, np.nan)
    mask = df["_return_pct"].abs() >= LABEL_THRESHOLD
    df = df[mask].copy()
    df["label"] = np.where(df["_return_pct"] > 0, 1, 0).astype(int)
    df.drop(columns=["_close_future", "_return_pct"], inplace=True)
    return df


def _global_date_split(df):
    dates = sorted(df["Date"].unique())
    n = len(dates)
    d1 = dates[int(n * TRAIN_RATIO)]
    d2 = dates[int(n * (TRAIN_RATIO + VAL_RATIO))]
    return (df[df["Date"] <  d1].copy(),
            df[(df["Date"] >= d1) & (df["Date"] < d2)].copy(),
            df[df["Date"] >= d2].copy())


def _news_proba(df):
    """Convert news_score (or pos/neg) to [P(DOWN), P(UP)] probabilities."""
    if "news_score" in df.columns:
        score = df["news_score"].fillna(0.0).values.astype(np.float32)
        p_up  = 1.0 / (1.0 + np.exp(-score * 3))
        return np.column_stack([1.0 - p_up, p_up])
    elif all(c in df.columns for c in ["news_positive", "news_negative"]):
        arr = df[["news_positive", "news_negative"]].fillna(0.5).values.astype(np.float32)
        return arr / arr.sum(axis=1, keepdims=True).clip(min=1e-9)
    return np.full((len(df), 2), 0.5)


def train():
    print("\n" + "="*55)
    print("  ENSEMBLE META-LEARNER - Training (FIXED v9)")
    print("="*55)

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("OK", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df = df.sort_values("Date").reset_index(drop=True)

    # FIX 4: shared feature pipeline (same as XGBoost/LSTM training)
    _p("i", "Engineering features ...")
    df = _engineer_features(df)

    # FIX 2: 5-day label with 0.5% threshold
    before = len(df)
    df = _apply_threshold_labels(df)
    _p("OK", f"Threshold filter: {before} → {len(df)} rows")

    _, val_df, test_df = _global_date_split(df)
    # FIX 3: use ALL of val set for meta-training (gives more meta-train data)
    meta_train_df = val_df.copy()
    _p("OK", f"Meta-train:{len(meta_train_df)}  Test:{len(test_df)}")

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
                    _p("OK", f"{name} probabilities computed for {len(subset_df)} rows")
                except Exception as e:
                    _p("!", f"{name} predict failed: {e}; uniform fill")
                    parts.append(np.full((len(subset_df), 2), 0.5))
            else:
                _p("!", f"{name} not loaded; uniform fill")
                parts.append(np.full((len(subset_df), 2), 0.5))
        parts.append(_news_proba(subset_df))
        return np.hstack(parts)   # shape: (n, 6) — [xgb_down, xgb_up, lstm_down, lstm_up, news_down, news_up]

    X_meta = _get_parts(meta_train_df)
    X_test = _get_parts(test_df)
    y_meta = meta_train_df["label"].values
    y_test = test_df["label"].values

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

    # Use meta predictions as both train and val for reporting
    metrics = evaluate_all(
        y_meta, y_pred_meta, y_proba_meta,
        y_meta, y_pred_meta, y_proba_meta,  # no separate val split now
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
        if "Direction" in test_df.columns:
            res["Direction"] = test_df["Direction"]
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
        _p("OK", f"Results -> {ENSEMBLE_RESULTS_PATH}")
    except Exception as e: _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()
