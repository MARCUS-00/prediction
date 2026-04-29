import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config.settings import (
    LSTM_MODEL_PATH, LSTM_SCALER_PATH, LSTM_FEATURES, SEQUENCE_LENGTH,
)
from models.lstm.model import LSTMClassifier


def load_lstm():
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(
            f"LSTM model not found at {LSTM_MODEL_PATH}. "
            f"Run: python models/lstm/train.py")

    state = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
    n_feat = state["lstm.weight_ih_l0"].shape[1]
    net = LSTMClassifier(n_feat)
    net.load_state_dict(state)
    net.eval()

    if not os.path.exists(LSTM_SCALER_PATH):
        raise FileNotFoundError(
            f"LSTM scaler not found at {LSTM_SCALER_PATH}.")
    with open(LSTM_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    feat_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_features.pkl")
    feats = list(LSTM_FEATURES)
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            feats = pickle.load(f)

    seqlen_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_seqlen.pkl")
    seq_len = SEQUENCE_LENGTH
    if os.path.exists(seqlen_path):
        with open(seqlen_path, "rb") as f:
            seq_len = pickle.load(f)

    if len(feats) != n_feat:
        raise RuntimeError(
            f"Feature-count mismatch: model expects {n_feat}, saved features = {len(feats)}")

    return {"net": net, "scaler": scaler, "n_features": n_feat,
            "feats": feats, "seq_len": seq_len}


def _build_inference_sequences(stock_df, scaler, feats, seq_len):
    df = stock_df.sort_values("Date").copy().reset_index(drop=True)
    if len(df) < seq_len:
        return np.empty((0, seq_len, len(feats)), dtype=np.float32)
    vals = np.nan_to_num(df[feats].values.astype(np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    vals = scaler.transform(vals)
    X = []
    for i in range(seq_len, len(df) + 1):
        X.append(vals[i - seq_len: i])
    if not X:
        return np.empty((0, seq_len, len(feats)), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def predict_proba(df, payload=None):
    if payload is None:
        payload = load_lstm()
    if len(df) == 0:
        return np.empty((0, 2))

    net     = payload["net"]
    scaler  = payload["scaler"]
    feats   = payload["feats"]
    seq_len = payload["seq_len"]

    df = df.copy()
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    df["__orig_idx"] = np.arange(len(df))
    out_probas = np.full((len(df), 2), 0.5, dtype=np.float32)

    for stock in df["Stock"].unique():
        sdf = df[df["Stock"] == stock].sort_values("Date")
        indices = sdf["__orig_idx"].values
        X = _build_inference_sequences(sdf, scaler, feats, seq_len)

        if len(X) == 0:
            continue

        with torch.no_grad():
            probas = F.softmax(
                net(torch.tensor(X, dtype=torch.float32)), dim=1
            ).numpy()

        n_pad = len(sdf) - len(probas)
        if n_pad > 0:
            pad = np.full((n_pad, 2), 0.5, dtype=np.float32)
            stock_probas = np.vstack([pad, probas]).astype(np.float32)
        else:
            stock_probas = probas[-len(sdf):].astype(np.float32)

        out_probas[indices] = stock_probas

    return out_probas
