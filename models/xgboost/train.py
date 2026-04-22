# =============================================================================
# models/xgboost/train.py  (FIXED v3)
#
# CRITICAL FIX: switched from per-stock temporal split to GLOBAL date split.
#
# Root cause of inflated 79% accuracy:
#   Per-stock split assigns different cutoff dates to each stock.
#   Cross-sectional features (CS_momentum_rank, CS_rsi_rank, CS_volume_rank)
#   are computed at merge time using ALL stocks on each date together.
#   When stock A's date 2022-09-15 is in TRAIN but stock B's same date is in VAL,
#   B's VAL rows have CS features that encode A's TRAIN information → leakage.
#   Result: artificially inflated 79% that collapses to 50% on true out-of-sample data.
#
# Fix: ALL stocks use the same global date cutoffs (D1, D2).
#   - Train: all rows with Date < D1
#   - Val:   all rows with D1 ≤ Date < D2
#   - Test:  all rows with Date ≥ D2
# This ensures CS features on any given date only mix rows from the same split.
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
    """
    Split by global calendar date, not per-stock index.
    Prevents cross-sectional feature leakage between train and val/test.
    """
    dates = sorted(df["Date"].unique())
    n = len(dates)
    d1 = dates[int(n * TRAIN_RATIO)]
    d2 = dates[int(n * (TRAIN_RATIO + VAL_RATIO))]
    train = df[df["Date"] <  d1].copy()
    val   = df[(df["Date"] >= d1) & (df["Date"] < d2)].copy()
    test  = df[df["Date"] >= d2].copy()
    _p("✓", f"Global date split — train<{d1}  val[{d1},{d2})  test≥{d2}")
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
    print("  XGBOOST — Training")
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
        _p("✓", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)

    # FIXED: global date split instead of per-stock split
    train_df, val_df, test_df = _global_date_split(df)
    _p("✓", f"Split — train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    X_train, y_train = _get_X(train_df), train_df["label"].values
    X_val,   y_val   = _get_X(val_df),   val_df["label"].values
    X_test,  y_test  = _get_X(test_df),  test_df["label"].values
    _p("i", f"Training on {X_train.shape[1]} features, {len(X_train)} rows ...")

    # Class imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}

    model = XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

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

    payload = {
        "model":         model,
        "feature_names": X_train.columns.tolist(),
        "metrics":       metrics,
    }
    try:
        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        with open(XGB_MODEL_PATH, "wb") as f: pickle.dump(payload, f)
        _p("✓", f"Model → {XGB_MODEL_PATH}")
    except Exception as e:
        _p("x", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date", "Stock", "Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(XGB_RESULTS_PATH, index=False)
        _p("✓", f"Results → {XGB_RESULTS_PATH}")
    except Exception as e:
        _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()