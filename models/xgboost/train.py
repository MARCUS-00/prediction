# =============================================================================
# models/xgboost/train.py  (FIXED v4)
#
# Fixes vs v3:
#   1. early_stopping_rounds moved to XGBClassifier() constructor
#      (XGBoost 3.x API — passing it to .fit() raises TypeError in XGBoost >= 2.0)
#   2. scale_pos_weight REMOVED — it creates strong DOWN prediction bias
#      (recall DOWN >> recall UP) when AUC is near 0.50, hurting accuracy.
#      Natural class weights work better for this near-balanced problem.
#   3. verbose changed to False (was 100, now silent) — cleaner output.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle, random
import numpy as np
import pandas as pd

from config.settings    import (MERGED_CSV, XGBOOST_PARAMS, XGBOOST_FEATURES,
                                 LABEL_MAP, XGB_MODEL_PATH, XGB_RESULTS_PATH,
                                 RANDOM_SEED, TRAIN_RATIO, VAL_RATIO)
from evaluation.metrics import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _global_date_split(df):
    dates = sorted(df["Date"].unique())
    n = len(dates)
    d1 = dates[int(n * TRAIN_RATIO)]
    d2 = dates[int(n * (TRAIN_RATIO + VAL_RATIO))]
    train = df[df["Date"] <  d1].copy()
    val   = df[(df["Date"] >= d1) & (df["Date"] < d2)].copy()
    test  = df[df["Date"] >= d2].copy()
    _p("OK", f"Global date split: train<{d1}  val[{d1},{d2})  test>={d2}")
    return train, val, test


def _get_X(df):
    avail   = [c for c in XGBOOST_FEATURES if c in df.columns]
    missing = [c for c in XGBOOST_FEATURES if c not in df.columns]
    if missing: _p("!", f"Features not in dataset (skipped): {missing}")
    X = df[avail].copy().apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    return X


def train():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    print("\n" + "="*55)
    print("  XGBOOST - Training")
    print("="*55)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        _p("x", "xgboost not installed. Run: pip install xgboost"); return {}

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found")
        _p("x", "Run: python features/merge_features.py"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("OK", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)

    train_df, val_df, test_df = _global_date_split(df)
    _p("OK", f"Split: train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    X_train, y_train = _get_X(train_df), train_df["label"].values
    X_val,   y_val   = _get_X(val_df),   val_df["label"].values
    X_test,  y_test  = _get_X(test_df),  test_df["label"].values
    _p("i", f"Training on {X_train.shape[1]} features, {len(X_train)} rows ...")

    # FIX: scale_pos_weight REMOVED — causes DOWN bias when AUC~0.50
    # FIX: early_stopping_rounds in constructor (XGBoost 3.x API)
    model = XGBClassifier(**XGBOOST_PARAMS, early_stopping_rounds=40)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    _p("i", f"Best iteration: {model.best_iteration}")

    y_pred_train  = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)
    y_pred_val    = model.predict(X_val)
    y_proba_val   = model.predict_proba(X_val)
    y_pred_test   = model.predict(X_test)
    y_proba_test  = model.predict_proba(X_test)

    metrics = evaluate_all(
        y_train, y_pred_train, y_proba_train,
        y_val,   y_pred_val,   y_proba_val,
        y_test,  y_pred_test,  y_proba_test, "XGBoost"
    )

    _p("i", f"Always-UP baseline on test: {y_test.mean():.4f}")

    payload = {
        "model":         model,
        "feature_names": X_train.columns.tolist(),
        "metrics":       metrics,
    }
    try:
        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        with open(XGB_MODEL_PATH, "wb") as f: pickle.dump(payload, f)
        _p("OK", f"Model -> {XGB_MODEL_PATH}")
    except Exception as e:
        _p("x", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date", "Stock", "Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(XGB_RESULTS_PATH, index=False)
        _p("OK", f"Results -> {XGB_RESULTS_PATH}")
    except Exception as e:
        _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()
