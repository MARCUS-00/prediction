"""
models/ensemble/predict.py
==========================
BUGS FIXED:
  BUG-3  Double-mapping of predicted label (int_to_idx applied twice).
         Now: classes_[argmax_idx] gives external label directly.
  BUG-1  Exposed prob_up as proba[:,2] (not proba[:,1]=P(FLAT)).
  TORCH  Lazy LSTM import so app works even if PyTorch is unavailable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd

from config.settings import META_MODEL_PATH, LABEL_MAP_INV, NEWS_CSV
from models.xgboost.predict import predict_proba as xgb_proba, load_xgb

NUM_CLASS = 3
FINBERT_SCORES_PATH = os.path.join(os.path.dirname(NEWS_CSV), "finbert_scores.csv")

_INT_TO_LABEL = {0: "DOWN", 1: "FLAT", 2: "UP"}
_EXT_TO_INT   = {-1: 0, 0: 1, 1: 2}


def _lazy_lstm():
    """Import LSTM predict lazily so torch errors don't crash the whole module."""
    try:
        from models.lstm.predict import predict_proba as lp, load_lstm as ll
        return lp, ll
    except Exception:
        return None, None


def load_meta():
    if not os.path.exists(META_MODEL_PATH):
        raise FileNotFoundError(
            f"Meta model not found at {META_MODEL_PATH}. "
            "Run: python models/ensemble/train_meta.py")
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
    n      = len(df)
    result = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if fb is None or fb.empty:
        return result
    merged        = df[["Date", "Stock"]].copy().reset_index(drop=True)
    merged["_idx"] = range(n)
    merged         = merged.merge(fb, on=["Date", "Stock"], how="left")
    for col_i, col in enumerate(["finbert_pos", "finbert_neg", "finbert_neu"]):
        if col in merged.columns:
            vals = merged[col].values.astype(float)
            mask = ~np.isnan(vals)
            result[merged.loc[mask, "_idx"].values, col_i] = vals[mask]
    return result


def predict_ensemble(df, xgb_payload=None, lstm_payload=None, meta_payload=None):
    if len(df) == 0:
        out = df.copy()
        for col in ("Predicted", "Confidence", "Direction_Label",
                    "prob_down", "prob_flat", "prob_up"):
            out[col] = []
        return out

    if xgb_payload is None:
        xgb_payload = load_xgb()

    # Lazy LSTM import
    lstm_proba_fn, load_lstm_fn = _lazy_lstm()
    if lstm_payload is None and load_lstm_fn is not None:
        try:
            lstm_payload = load_lstm_fn()
        except Exception:
            lstm_payload = None

    if meta_payload is None:
        try:
            meta_payload = load_meta()
        except Exception:
            meta_payload = None

    n = len(df)

    # ── XGBoost (N, 3): [P(DOWN), P(FLAT), P(UP)] ────────────────────────
    try:
        xp = xgb_proba(df, xgb_payload)
        if xp.shape != (n, NUM_CLASS):
            xp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS, dtype=np.float32)
    except Exception:
        xp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS, dtype=np.float32)

    # ── LSTM (N, 3) ───────────────────────────────────────────────────────
    lp = None
    if lstm_payload is not None and lstm_proba_fn is not None:
        try:
            lp = lstm_proba_fn(df, lstm_payload)
            if lp.shape != (n, NUM_CLASS):
                lp = np.full((n, NUM_CLASS), 1.0 / NUM_CLASS, dtype=np.float32)
        except Exception:
            lp = None

    # ── FinBERT (N, 3) ────────────────────────────────────────────────────
    fb_scores = _load_finbert_scores()
    fp        = _get_finbert_features(df, fb_scores)

    # ── Final probabilities ───────────────────────────────────────────────
    if meta_payload is not None and lp is not None:
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

    # ── Decode ────────────────────────────────────────────────────────────
    argmax_idx = proba.argmax(axis=1)   # 0/1/2

    if meta_payload is not None:
        classes    = meta_payload["meta_model"].classes_   # [-1, 0, 1]
        ext_labels = np.array([int(classes[i]) for i in argmax_idx])
        # FIX: ext_labels is already the external label {-1,0,1};
        #      map to internal index only for storing in "Predicted"
        predicted  = [_EXT_TO_INT[int(l)] for l in ext_labels]
        dir_labels = [_INT_TO_LABEL[_EXT_TO_INT[int(l)]] for l in ext_labels]
    else:
        predicted  = list(argmax_idx)
        dir_labels = [_INT_TO_LABEL[int(i)] for i in argmax_idx]

    out                    = df.copy()
    out["Predicted"]       = predicted        # internal {0,1,2}
    out["Confidence"]      = proba.max(axis=1)
    out["Direction_Label"] = dir_labels        # DOWN/FLAT/UP strings
    out["prob_down"]       = proba[:, 0]
    out["prob_flat"]       = proba[:, 1]
    out["prob_up"]         = proba[:, 2]      # FIX: col 2, not col 1
    return out
