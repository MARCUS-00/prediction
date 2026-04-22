# =============================================================================
# models/lstm/predict.py  (FIXED)
#
# FIXED: _build_sequences() now requires `feats` argument (positional).
#        Load feats from saved lstm_features.pkl alongside the model.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import torch
import torch.nn.functional as F

from config.settings   import LSTM_MODEL_PATH, LSTM_SCALER_PATH, LSTM_FEATURES, SEQUENCE_LENGTH
from models.lstm.model import LSTMClassifier
from models.lstm.train import _build_sequences


def load_lstm():
    try:
        state  = torch.load(LSTM_MODEL_PATH, map_location="cpu")
        n_feat = state["lstm.weight_ih_l0"].shape[1]
        net    = LSTMClassifier(n_feat)
        net.load_state_dict(state); net.eval()

        with open(LSTM_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        # Load saved feature list (written by train.py)
        feat_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_features.pkl")
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as f:
                feats = pickle.load(f)
        else:
            feats = LSTM_FEATURES  # fallback to settings

        return {"net": net, "scaler": scaler, "n_features": n_feat, "feats": feats}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LSTM model not found at {LSTM_MODEL_PATH}. Run models/lstm/train.py")
    except Exception as e:
        raise RuntimeError(f"LSTM load failed: {e}")


def predict_proba(df, payload=None):
    if payload is None:
        payload = load_lstm()

    net    = payload["net"]
    scaler = payload["scaler"]
    feats  = payload.get("feats", LSTM_FEATURES)

    tmp = df.copy()
    tmp["__orig_idx"] = np.arange(len(tmp))
    if "label" not in tmp.columns:
        tmp["label"] = 0

    out_probas = np.full((len(tmp), 2), 0.5)

    for stock in tmp["Stock"].unique():
        sdf     = tmp[tmp["Stock"] == stock].sort_values("Date")
        indices = sdf["__orig_idx"].values

        # FIXED: pass feats explicitly
        X, _ = _build_sequences(sdf, scaler=scaler, feats=feats)

        if len(X) > 0:
            with torch.no_grad():
                probas = F.softmax(
                    net(torch.tensor(X, dtype=torch.float32)), dim=1
                ).numpy()
            # Pad rows that couldn't form a full sequence (first SEQUENCE_LENGTH rows)
            pad = np.full((len(sdf) - len(probas), 2), 0.5)
            stock_probas = np.vstack([pad, probas])
        else:
            stock_probas = np.full((len(sdf), 2), 0.5)

        out_probas[indices] = stock_probas

    return out_probas