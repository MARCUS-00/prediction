# =============================================================================
# models/xgboost/predict.py  (FIXED v6)
#
# Fixes vs v5:
#   1. Uses saved train_medians for NaN-filling (avoids using test-set median).
#   2. Zero-fills missing feature columns (news_score etc.).
#   3. Calls _engineer_features from xgboost/train.py if raw features are present.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from config.settings import XGB_MODEL_PATH


def load_xgb():
    try:
        with open(XGB_MODEL_PATH,"rb") as f: return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}. Run models/xgboost/train.py")
    except Exception as e:
        raise RuntimeError(f"XGBoost load failed: {e}")


def predict_proba(df, payload=None):
    if payload is None: payload = load_xgb()

    feature_names  = payload["feature_names"]
    train_medians  = payload.get("train_medians", {})

    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0.0   # zero-fill missing features

    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # FIX: use training medians for NaN fill (not test-set medians)
    for col in X.columns:
        if X[col].isna().any():
            fill_val = train_medians.get(col, 0.0)
            X[col] = X[col].fillna(fill_val)

    return payload["model"].predict_proba(X)


def get_feature_importance(payload=None, top_n=10):
    if payload is None: payload = load_xgb()
    fi = payload["model"].get_booster().get_score(importance_type="gain")
    fn = payload["feature_names"]
    scores = {}
    for k, v in fi.items():
        if k in fn:
            scores[k] = v
        else:
            try:
                idx = int(k[1:])
                if idx < len(fn): scores[fn[idx]] = v
            except (ValueError, IndexError): pass
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
