# =============================================================================
# models/lstm/predict.py  (FIXED v9)
#
# Fixes vs v8:
#   1. BUG FIX: _build_sequences() called with positional seq_len argument.
#      v8 called _build_sequences(sdf, scaler=scaler, feats=feats) with only
#      3 args — crashing because seq_len had no default in v8's signature.
#      v9 adds seq_len=SEQ_LEN default in train.py AND passes it explicitly
#      here for clarity.
#
#   2. Load seq_len from lstm_seqlen.pkl (saved by train.py) instead of
#      importing SEQUENCE_LENGTH from settings. This ensures the correct
#      window is used even if settings.py was modified after training.
#
#   3. Feature list always loaded from lstm_features.pkl; settings fallback
#      now also adds return_vs_sector, news_rolling_3d.
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
        state  = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
        n_feat = state["lstm.weight_ih_l0"].shape[1]
        net    = LSTMClassifier(n_feat)
        net.load_state_dict(state); net.eval()

        with open(LSTM_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        feat_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_features.pkl")
        feats = LSTM_FEATURES  # fallback
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as f:
                feats = pickle.load(f)

        # FIX 2: Load the actual seq_len used at training time
        seqlen_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_seqlen.pkl")
        seq_len = SEQUENCE_LENGTH  # fallback
        if os.path.exists(seqlen_path):
            with open(seqlen_path, "rb") as f:
                seq_len = pickle.load(f)

        return {
            "net": net,
            "scaler": scaler,
            "n_features": n_feat,
            "feats": feats,
            "seq_len": seq_len,   # FIX 2: stored so ensemble can pass it
        }
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LSTM model not found at {LSTM_MODEL_PATH}. Run models/lstm/train.py")
    except Exception as e:
        raise RuntimeError(f"LSTM load failed: {e}")


def predict_proba(df, payload=None):
    if payload is None:
        payload = load_lstm()

    net     = payload["net"]
    scaler  = payload["scaler"]
    feats   = payload.get("feats", LSTM_FEATURES)
    seq_len = payload.get("seq_len", SEQUENCE_LENGTH)  # FIX 2

    # Zero-fill any missing feature columns
    df = df.copy()
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    df["__orig_idx"] = np.arange(len(df))
    if "label" not in df.columns:
        df["label"] = 0

    out_probas = np.full((len(df), 2), 0.5)

    for stock in df["Stock"].unique():
        sdf     = df[df["Stock"] == stock].sort_values("Date")
        indices = sdf["__orig_idx"].values

        # FIX 1: Pass seq_len explicitly (v9 signature has default but explicit is safer)
        X, _ = _build_sequences(sdf, scaler=scaler, feats=feats, seq_len=seq_len)

        if len(X) > 0:
            with torch.no_grad():
                probas = F.softmax(
                    net(torch.tensor(X, dtype=torch.float32)), dim=1
                ).numpy()
            # Pad rows at start of stock history that have no full sequence
            pad = np.full((len(sdf) - len(probas), 2), 0.5)
            stock_probas = np.vstack([pad, probas])
        else:
            stock_probas = np.full((len(sdf), 2), 0.5)

        out_probas[indices] = stock_probas

    return out_probas
