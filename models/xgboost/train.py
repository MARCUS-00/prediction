# =============================================================================
# models/xgboost/train.py  (FIXED v5)
#
# Critical fixes vs v4:
#   1. THRESHOLD-BASED LABELING: rows where |return_pct| < 0.5% are DROPPED.
#      This removes ~30% ambiguous noise rows, making the signal much cleaner.
#      return_pct = (Close[t+1] - Close[t]) / Close[t]
#      return > +0.5% → keep as UP (1), return < -0.5% → keep as DOWN (0)
#   2. XGBoost params upgraded: n_estimators=500, max_depth=6, lr=0.03,
#      colsample_bytree=0.8, min_child_weight=5 (from settings.py).
#   3. early_stopping_rounds=50 (was 40) — more patience for larger model.
#   4. verbose=100 restored so training progress is visible.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle, random
import numpy as np
import pandas as pd

from config.settings    import (MERGED_CSV, XGBOOST_PARAMS, XGBOOST_FEATURES,
                                 LABEL_MAP, LABEL_THRESHOLD, XGB_MODEL_PATH,
                                 XGB_RESULTS_PATH, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO)
from evaluation.metrics import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _apply_threshold_labels(df):
    """
    FIX: Threshold-based labeling.
    Compute actual next-day return and keep only rows where
    |return_pct| >= LABEL_THRESHOLD (0.5%). Drop ambiguous rows.
    """
    df = df.copy().sort_values(["Stock", "Date"]).reset_index(drop=True)
    df["_close_next"] = df.groupby("Stock")["Close"].shift(-1)
    df["_return_pct"] = (df["_close_next"] - df["Close"]) / df["Close"].replace(0, np.nan)

    # Keep only rows with a clear directional move
    mask = df["_return_pct"].abs() >= LABEL_THRESHOLD
    df = df[mask].copy()

    # Assign label: UP if positive, DOWN if negative
    df["label"] = np.where(df["_return_pct"] > 0, 1, 0)
    df.drop(columns=["_close_next", "_return_pct"], inplace=True)
    return df


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
    print("  XGBOOST - Training (FIXED v5)")
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

    df = df.sort_values("Date").reset_index(drop=True)

    # FIX: Apply threshold-based labeling BEFORE split
    before = len(df)
    df = _apply_threshold_labels(df)
    after = len(df)
    _p("OK", f"Threshold filter: {before} → {after} rows "
             f"({(before-after)/before*100:.1f}% dropped as noise)")
    _p("OK", f"Label distribution: UP={df['label'].sum()} DOWN={(df['label']==0).sum()}")

    train_df, val_df, test_df = _global_date_split(df)
    _p("OK", f"Split: train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    X_train, y_train = _get_X(train_df), train_df["label"].values
    X_val,   y_val   = _get_X(val_df),   val_df["label"].values
    X_test,  y_test  = _get_X(test_df),  test_df["label"].values
    _p("i", f"Training on {X_train.shape[1]} features, {len(X_train)} rows ...")

    model = XGBClassifier(**XGBOOST_PARAMS, early_stopping_rounds=50)
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
