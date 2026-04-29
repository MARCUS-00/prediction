import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd

from config.settings import XGB_MODEL_PATH


def load_xgb():
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found at {XGB_MODEL_PATH}. "
            f"Run: python models/xgboost/train.py")
    with open(XGB_MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    required = ["model", "feature_names", "train_medians"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"XGBoost payload missing keys: {missing}")
    return payload


def _build_feature_matrix(df, payload):
    feature_names = payload["feature_names"]
    train_medians = payload.get("train_medians", {})

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = train_medians.get(col, 0.0)

    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    fill_series = pd.Series({c: train_medians.get(c, 0.0) for c in X.columns})
    X = X.fillna(fill_series).fillna(0.0)
    return X


def predict_proba(df, payload=None):
    if payload is None:
        payload = load_xgb()
    if len(df) == 0:
        return np.empty((0, 2))
    X = _build_feature_matrix(df, payload)
    return payload["model"].predict_proba(X)


def predict_label(df, payload=None, threshold=None):
    if payload is None:
        payload = load_xgb()
    if threshold is None:
        threshold = payload.get("threshold", 0.5)
    proba = predict_proba(df, payload)
    return (proba[:, 1] >= threshold).astype(int), proba


def get_feature_importance(payload=None, top_n=10):
    if payload is None:
        payload = load_xgb()
    base = payload.get("base_model")
    feature_names = payload["feature_names"]
    if base is None:
        return {}
    try:
        importances = base.feature_importances_
        pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        return {name: float(score) for name, score in pairs}
    except Exception:
        return {}
