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
    avail = [c for c in payload["feature_names"] if c in df.columns]
    X = df[avail].copy().apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    return payload["model"].predict_proba(X)


def get_feature_importance(payload=None, top_n=10):
    if payload is None: payload = load_xgb()
    fi = payload["model"].get_booster().get_score(importance_type="weight")
    fn = payload["feature_names"]
    scores = {}
    for k,v in fi.items():
        if k in fn:
            scores[k] = v
        else:
            try:
                idx = int(k[1:])
                if idx < len(fn): scores[fn[idx]] = v
            except (ValueError, IndexError): pass
    return dict(sorted(scores.items(), key=lambda x:x[1], reverse=True)[:top_n])