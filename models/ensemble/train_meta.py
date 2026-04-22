import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV   # probability calibration

from config.settings         import (MERGED_CSV, LABEL_MAP, META_MODEL_PATH,
                                      ENSEMBLE_RESULTS_PATH, TRAIN_RATIO, VAL_RATIO,
                                      RANDOM_SEED)
from models.xgboost.predict  import predict_proba as xgb_proba, load_xgb
from models.lstm.predict     import predict_proba  as lstm_proba, load_lstm
from evaluation.metrics      import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _news_proba(df):
    """
    Derive a 2-class probability from sentiment score.
    Uses news_score = positive - negative if available; falls back to 0.5/0.5.
    """
    if "news_score" in df.columns:
        score = df["news_score"].fillna(0.0).values.astype(np.float32)
        # Sigmoid to convert score in (-1,1) → probability of UP
        p_up  = 1.0 / (1.0 + np.exp(-score * 3))    # scale 3 gives reasonable spread
        p_dn  = 1.0 - p_up
        return np.column_stack([p_dn, p_up])
    elif all(c in df.columns for c in ["news_positive", "news_negative"]):
        arr = df[["news_positive", "news_negative"]].fillna(0.5).values.astype(np.float32)
        row_sums = arr.sum(axis=1, keepdims=True).clip(min=1e-9)
        return arr / row_sums
    return np.full((len(df), 2), 0.5)


def train():
    print("\n" + "="*55)
    print("  ENSEMBLE META-LEARNER — Training")
    print("="*55)

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found")
        _p("x", "Run: python features/merge_features.py")
        return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("✓", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}")
        return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    val_dfs, test_dfs = [], []
    for stock in df["Stock"].unique():
        sdf = df[df["Stock"] == stock].sort_values("Date").reset_index(drop=True)
        n = len(sdf)
        if n < 30:
            continue
        t = int(n * TRAIN_RATIO)
        v = int(n * (TRAIN_RATIO + VAL_RATIO))
        val_dfs.append(sdf.iloc[t:v])
        test_dfs.append(sdf.iloc[v:])

    val_df  = pd.concat(val_dfs,  ignore_index=True) if val_dfs  else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    # BUG FIX: meta-learner was trained AND evaluated on val_df only.
    # Correct stacking: train meta on val, evaluate on held-out test.
    _p("✓", f"Meta-train (val): {len(val_df)}   Meta-test: {len(test_df)}")

    # ── Load base models ──────────────────────────────────────────────────────
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

    # ── Train meta-learner on val; predict on test ────────────────────────────
    _p("i", "Training Logistic Regression meta-learner on val-set stacked features...")
    base_lr = LogisticRegression(
        max_iter=2000, random_state=RANDOM_SEED, C=0.5,
        solver="lbfgs", class_weight="balanced",
    )
    # Calibrate to improve probability estimates
    meta = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")
    meta.fit(X_val, y_val)

    y_pred_val   = meta.predict(X_val)
    y_proba_val  = meta.predict_proba(X_val)
    y_pred_test  = meta.predict(X_test)
    y_proba_test = meta.predict_proba(X_test)

    # BUG FIX: was passing y_val as "train" AND "val" with identical preds.
    # Now val is train-equivalent; test is the proper out-of-sample set.
    metrics = evaluate_all(
        y_val,  y_pred_val,  y_proba_val,
        y_val,  y_pred_val,  y_proba_val,    # val == meta train (reported as "train")
        y_test, y_pred_test, y_proba_test,
        "Ensemble"
    )

    payload = {"meta_model": meta, "metrics": metrics}
    try:
        os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
        with open(META_MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        _p("✓", f"Model → {META_MODEL_PATH}")
    except Exception as e:
        _p("x", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date", "Stock", "Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
        _p("✓", f"Results → {ENSEMBLE_RESULTS_PATH}")
    except Exception as e:
        _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()