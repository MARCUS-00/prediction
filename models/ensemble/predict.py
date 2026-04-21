import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from config.settings         import META_MODEL_PATH, LABEL_MAP_INV
from models.xgboost.predict  import predict_proba as xgb_proba, load_xgb
from models.lstm.predict     import predict_proba  as lstm_proba, load_lstm
from models.ensemble.train_meta import _news_proba


def load_meta():
    try:
        with open(META_MODEL_PATH,"rb") as f: return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Meta model not found at {META_MODEL_PATH}. Run models/ensemble/train_meta.py")
    except Exception as e:
        raise RuntimeError(f"Meta load failed: {e}")


def predict_ensemble(df, xgb_payload=None, lstm_payload=None, meta_payload=None):
    if xgb_payload  is None: xgb_payload  = load_xgb()
    if lstm_payload is None: lstm_payload = load_lstm()
    if meta_payload is None: meta_payload = load_meta()

    parts = []
    for fn, payload in [(xgb_proba,xgb_payload),(lstm_proba,lstm_payload)]:
        try:   parts.append(fn(df, payload))
        except Exception: parts.append(np.full((len(df),3), 1/3))
    parts.append(_news_proba(df))

    proba = meta_payload["meta_model"].predict_proba(np.hstack(parts))
    preds = proba.argmax(axis=1)
    out   = df.copy()
    out["Predicted"]       = preds
    out["Confidence"]      = proba.max(axis=1)
    out["Direction_Label"] = [LABEL_MAP_INV[p] for p in preds]
    return out