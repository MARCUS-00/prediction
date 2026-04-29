import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd

from config.settings import META_MODEL_PATH, LABEL_MAP_INV
from models.xgboost.predict import predict_proba as xgb_proba, load_xgb
from models.lstm.predict    import predict_proba as lstm_proba, load_lstm


def load_meta():
    if not os.path.exists(META_MODEL_PATH):
        raise FileNotFoundError(
            f"Meta model not found at {META_MODEL_PATH}. "
            f"Run: python models/ensemble/train_meta.py")
    with open(META_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_ensemble(df, xgb_payload=None, lstm_payload=None, meta_payload=None):
    if len(df) == 0:
        out = df.copy()
        out["Predicted"] = []
        out["Confidence"] = []
        out["Direction_Label"] = []
        return out

    if xgb_payload is None:
        xgb_payload = load_xgb()

    try:
        if lstm_payload is None:
            lstm_payload = load_lstm()
    except Exception:
        lstm_payload = None

    try:
        if meta_payload is None:
            meta_payload = load_meta()
    except Exception:
        meta_payload = None

    n = len(df)

    try:
        xp = xgb_proba(df, xgb_payload)
        if xp.shape != (n, 2):
            xp = np.full((n, 2), 0.5)
    except Exception:
        xp = np.full((n, 2), 0.5)

    if lstm_payload is not None:
        try:
            lp = lstm_proba(df, lstm_payload)
            if lp.shape != (n, 2):
                lp = np.full((n, 2), 0.5)
        except Exception:
            lp = np.full((n, 2), 0.5)
    else:
        lp = None

    if meta_payload is not None and lp is not None:
        meta_X = np.column_stack([xp[:, 0], xp[:, 1], lp[:, 0], lp[:, 1]])
        proba = meta_payload["meta_model"].predict_proba(meta_X)
    elif lp is not None:
        proba = 0.5 * xp + 0.5 * lp
    else:
        proba = xp

    preds = proba.argmax(axis=1)
    out = df.copy()
    out["Predicted"]       = preds
    out["Confidence"]      = proba.max(axis=1)
    out["Direction_Label"] = [LABEL_MAP_INV[int(p)] for p in preds]
    return out
