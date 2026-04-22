# =============================================================================
# models/ensemble/train_meta.py  (FIXED v3)
#
# Same global-date-split fix applied.
# Also fixes ensemble/predict.py reference to num_classes=3 fallback.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from config.settings         import (MERGED_CSV, LABEL_MAP, META_MODEL_PATH,
                                      ENSEMBLE_RESULTS_PATH, TRAIN_RATIO, VAL_RATIO,
                                      RANDOM_SEED)
from models.xgboost.predict  import predict_proba as xgb_proba, load_xgb
from models.lstm.predict     import predict_proba  as lstm_proba, load_lstm
from evaluation.metrics      import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _global_date_split(df):
    dates = sorted(df["Date"].unique())
    n = len(dates)
    d1 = dates[int(n * TRAIN_RATIO)]
    d2 = dates[int(n * (TRAIN_RATIO + VAL_RATIO))]
    return (df[df["Date"] <  d1].copy(),
            df[(df["Date"] >= d1) & (df["Date"] < d2)].copy(),
            df[df["Date"] >= d2].copy())


def _news_proba(df):
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
    print("  ENSEMBLE META-LEARNER — Training")
    print("="*55)

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("✓", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)

    # FIXED: global date split
    _, val_df, test_df = _global_date_split(df)
    _p("✓", f"Meta-train (val): {len(val_df)}   Meta-test: {len(test_df)}")

    try:    xgb  = load_xgb()
    except Exception as e: _p("!", f"XGBoost load failed: {e}"); xgb = None

    try:    lstm = load_lstm()
    except Exception as e: _p("!", f"LSTM load failed: {e}"); lstm = None

    def _get_parts(subset_df):
        parts = []
        for name, fn, payload in [("XGBoost", xgb_proba, xgb), ("LSTM", lstm_proba, lstm)]:
            if payload:
                try:
                    parts.append(fn(subset_df, payload))
                    _p("✓", f"{name} probabilities computed for {len(subset_df)} rows")
                except Exception as e:
                    _p("!", f"{name} predict failed: {e} — uniform fill")
                    parts.append(np.full((len(subset_df), 2), 0.5))
            else:
                parts.append(np.full((len(subset_df), 2), 0.5))
        parts.append(_news_proba(subset_df))
        return np.hstack(parts)

    X_val  = _get_parts(val_df)
    X_test = _get_parts(test_df)
    y_val  = val_df["label"].values
    y_test = test_df["label"].values

    _p("i", "Training Logistic Regression meta-learner...")
    base_lr = LogisticRegression(
        max_iter=2000, random_state=RANDOM_SEED, C=0.5,
        solver="lbfgs", class_weight="balanced",
    )
    meta = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")
    meta.fit(X_val, y_val)

    y_pred_val   = meta.predict(X_val)
    y_proba_val  = meta.predict_proba(X_val)
    y_pred_test  = meta.predict(X_test)
    y_proba_test = meta.predict_proba(X_test)

    metrics = evaluate_all(
        y_val,  y_pred_val,  y_proba_val,
        y_val,  y_pred_val,  y_proba_val,
        y_test, y_pred_test, y_proba_test,
        "Ensemble"
    )

    payload = {"meta_model": meta, "metrics": metrics}
    try:
        os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
        with open(META_MODEL_PATH, "wb") as f: pickle.dump(payload, f)
        _p("✓", f"Model → {META_MODEL_PATH}")
    except Exception as e: _p("x", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date", "Stock", "Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
        _p("✓", f"Results → {ENSEMBLE_RESULTS_PATH}")
    except Exception as e: _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()