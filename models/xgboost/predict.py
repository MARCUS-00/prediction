"""
models/xgboost/predict.py
=========================
Inference helpers for the 3-class XGBoost model.

BUGS FIXED:
  BUG-1  predict_label() used binary threshold on proba[:,1] = P(FLAT), not P(UP).
         Now uses argmax over 3 columns + INT→EXT mapping.
  BUG-2  get_feature_importance() had no guard when base_model is missing.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd

from config.settings import XGB_MODEL_PATH

_INT_TO_EXT = {0: -1, 1: 0, 2: 1}   # internal {0,1,2} → external {-1,0,1}
NUM_CLASS   = 3


def load_xgb():
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found at {XGB_MODEL_PATH}. "
            "Run: python models/xgboost/train.py")
    with open(XGB_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    required = ["model", "feature_names", "train_medians"]
    missing  = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"XGBoost payload missing keys: {missing}")
    return payload


def _build_feature_matrix(df, payload):
    feature_names = payload["feature_names"]
    train_medians = payload.get("train_medians", {})

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else train_medians.get(col, 0.0)

    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    fill = pd.Series({c: train_medians.get(c, 0.0) for c in X.columns})
    X    = X.fillna(fill).fillna(0.0)
    return X


def predict_proba(df, payload=None):
    """Return ndarray shape (N, 3): [P(DOWN), P(FLAT), P(UP)]."""
    if payload is None:
        payload = load_xgb()
    if len(df) == 0:
        return np.empty((0, NUM_CLASS), dtype=np.float32)
    X = _build_feature_matrix(df, payload)
    return payload["model"].predict_proba(X)


def predict_label(df, payload=None):
    """
    Return (ext_labels, proba).
    ext_labels ∈ {-1=DOWN, 0=FLAT, 1=UP}.

    FIX: was using binary threshold on proba[:,1]=P(FLAT).
    Now: argmax over all 3 classes → map to external label space.
    """
    if payload is None:
        payload = load_xgb()
    proba      = predict_proba(df, payload)
    int_labels = proba.argmax(axis=1)
    ext_labels = np.vectorize(_INT_TO_EXT.get)(int_labels)
    return ext_labels, proba


def get_feature_importance(payload=None, top_n=10):
    if payload is None:
        payload = load_xgb()
    base          = payload.get("base_model")
    feature_names = payload["feature_names"]
    if base is None:
        return {}
    try:
        importances = base.feature_importances_
        pairs = sorted(zip(feature_names, importances),
                       key=lambda x: x[1], reverse=True)[:top_n]
        return {name: float(score) for name, score in pairs}
    except Exception:
        return {}
