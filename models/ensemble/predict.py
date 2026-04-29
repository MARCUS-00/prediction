import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd

from config.settings import META_MODEL_PATH, LABEL_MAP_INV, NEWS_CSV
from models.xgboost.predict import predict_proba as xgb_proba, load_xgb
from models.lstm.predict    import predict_proba as lstm_proba, load_lstm

NUM_CLASS = 3
FINBERT_SCORES_PATH = os.path.join(os.path.dirname(NEWS_CSV), "finbert_scores.csv")


def load_meta():
    if not os.path.exists(META_MODEL_PATH):
        raise FileNotFoundError(
            f"Meta model not found at {META_MODEL_PATH}. "
            f"Run: python models/ensemble/train_meta.py")
    with open(META_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_finbert_scores():
    if not os.path.exists(FINBERT_SCORES_PATH):
        return None
    fb = pd.read_csv(FINBERT_SCORES_PATH, parse_dates=["Date"])
    fb = fb.groupby(["Date", "Stock"], as_index=False)[
        ["finbert_pos", "finbert_neg", "finbert_neu"]].mean()
    return fb


def _get_finbert_features(df, fb):
    n = len(df)
    result = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if fb is None or fb.empty:
        return result
    merged = df[["Date", "Stock"]].copy().reset_index(drop=True)
    merged["_idx"] = range(n)
    merged = merged.merge(fb, on=["Date", "Stock"], how="left")
    for col_i, col in enumerate(["finbert_pos", "finbert_neg", "finbert_neu"]):
        if col in merged.columns:
            vals = merged[col].values.astype(float)
            mask = ~np.isnan(vals)
            result[merged.loc[mask, "_idx"].values, col_i] = vals[mask]
    return result


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

    # XGBoost — expect (n, 3)
    try:
        xp = xgb_proba(df, xgb_payload)
        if xp.shape != (n, NUM_CLASS):
            xp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS)
    except Exception:
        xp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS)

    # LSTM — expect (n, 3)
    if lstm_payload is not None:
        try:
            lp = lstm_proba(df, lstm_payload)
            if lp.shape != (n, NUM_CLASS):
                lp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS)
        except Exception:
            lp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS)
    else:
        lp = None

    # FinBERT — (n, 3)
    fb_scores = _load_finbert_scores()
    fp = _get_finbert_features(df, fb_scores)  # always (n, 3)

    if meta_payload is not None and lp is not None:
        # 9 features: xgb(3) + lstm(3) + finbert(3)
        meta_X = np.column_stack([
            xp[:, 0], xp[:, 1], xp[:, 2],
            lp[:, 0], lp[:, 1], lp[:, 2],
            fp[:, 0], fp[:, 1], fp[:, 2],
        ])
        proba = meta_payload["meta_model"].predict_proba(meta_X)
    elif lp is not None:
        proba = (xp + lp) / 2.0
    else:
        proba = xp

    preds = proba.argmax(axis=1)
    # meta_model was trained with labels {-1, 0, 1}; classes_ maps to internal indices
    if meta_payload is not None:
        classes = meta_payload["meta_model"].classes_
        pred_labels = [classes[p] for p in preds]
        # convert {-1,0,1} → internal index {0,1,2} for LABEL_MAP_INV
        int_to_idx = {-1: 0, 0: 1, 1: 2}
        direction_labels = [LABEL_MAP_INV[int_to_idx[int(l)]] for l in pred_labels]
        predicted = [int_to_idx[int(l)] for l in pred_labels]
    else:
        predicted = list(preds)
        direction_labels = [LABEL_MAP_INV[int(p)] for p in preds]

    out = df.copy()
    out["Predicted"]       = predicted
    out["Confidence"]      = proba.max(axis=1)
    out["Direction_Label"] = direction_labels
    return out