import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import pickle
import numpy as np
import pandas as pd
import torch

from config.settings import (
    LSTM_MODEL_PATH, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT, SEQUENCE_LENGTH,
)
from models.lstm.model import LSTMClassifier

log = logging.getLogger("lstm_predict")

NUM_CLASS = 3   # DOWN / FLAT / UP


def load_lstm():
    """
    Load the trained LSTM.  Architecture is sourced from:
      1. Keys inside the checkpoint dict  (if saved by train.py with full payload)
      2. Companion .pkl sidecar files     (legacy saves)
      3. settings.py constants            (final fallback)
    This guarantees the loaded model always matches whatever was trained.
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(
            f"LSTM model not found at {LSTM_MODEL_PATH}. "
            "Run: python models/lstm/train.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)

    model_dir = os.path.dirname(LSTM_MODEL_PATH)

    # ── 1. State dict ────────────────────────────────────────────────────────
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
        # If the top-level dict IS the state dict (all keys are tensor params),
        # use it directly.
        if not isinstance(next(iter(state.values()), None), torch.Tensor):
            state = checkpoint   # fallback: entire checkpoint is the state dict
    else:
        state = checkpoint

    # ── 2. Feature columns ───────────────────────────────────────────────────
    feature_cols = []
    if isinstance(checkpoint, dict) and "feature_cols" in checkpoint:
        feature_cols = checkpoint["feature_cols"]
    else:
        feat_path = os.path.join(model_dir, "lstm_features.pkl")
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as f:
                feature_cols = pickle.load(f)

    # ── 3. Scaler ────────────────────────────────────────────────────────────
    scaler = None
    if isinstance(checkpoint, dict) and "scaler" in checkpoint:
        scaler = checkpoint["scaler"]
    else:
        scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")
        if os.path.exists(scaler_path):
            import joblib
            scaler = joblib.load(scaler_path)

    # ── 4. Sequence length ───────────────────────────────────────────────────
    seq_len = SEQUENCE_LENGTH
    if isinstance(checkpoint, dict) and "seq_len" in checkpoint:
        seq_len = checkpoint["seq_len"]
    else:
        seqlen_path = os.path.join(model_dir, "lstm_seqlen.pkl")
        if os.path.exists(seqlen_path):
            with open(seqlen_path, "rb") as f:
                seq_len = pickle.load(f)

    # ── 5. Architecture from checkpoint or settings ──────────────────────────
    hidden_size = LSTM_HIDDEN
    num_layers  = LSTM_LAYERS
    dropout     = LSTM_DROPOUT

    if isinstance(checkpoint, dict):
        hidden_size = checkpoint.get("hidden_size", hidden_size)
        num_layers  = checkpoint.get("num_layers",  num_layers)
        dropout     = checkpoint.get("dropout",      dropout)

    # Derive input_size from saved weights to avoid any mismatch
    # lstm.weight_ih_l0 has shape (4*hidden, input_size)
    wih_key = "lstm.weight_ih_l0"
    if wih_key in state:
        input_size = state[wih_key].shape[1]
    elif feature_cols:
        input_size = len(feature_cols)
    else:
        input_size = 25   # last-resort default
        log.warning("Cannot determine LSTM input_size; defaulting to 25")

    log.info(
        f"Loading LSTMClassifier: input={input_size}  hidden={hidden_size}  "
        f"layers={num_layers}  dropout={dropout}  num_classes={NUM_CLASS}"
    )

    net = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=NUM_CLASS,
    ).to(device)

    # strict=False lets us tolerate minor key mismatches from legacy saves
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"Missing keys in state_dict (will be random): {missing}")
    if unexpected:
        log.warning(f"Unexpected keys in state_dict (ignored): {unexpected}")

    net.eval()

    # Attach metadata for downstream callers
    net.scaler       = scaler
    net.feature_cols = feature_cols
    net.seq_len      = seq_len

    log.info(f"LSTM loaded from {LSTM_MODEL_PATH} on {device}")
    return net


def predict_proba(df_or_tensor, payload=None):
    """
    Accept either:
      • a numpy array / torch.Tensor of shape (N, seq_len, input_size)  [batch mode]
      • a pandas DataFrame with feature columns  [single-stock mode]

    Returns ndarray shape (N, 3): [P(DOWN), P(FLAT), P(UP)]
    """
    model = payload if payload is not None else load_lstm()
    device = next(model.parameters()).device
    model.eval()

    # ── Convert DataFrame → tensor ───────────────────────────────────────────
    if isinstance(df_or_tensor, pd.DataFrame):
        return predict_single(df_or_tensor, model)

    # ── Tensor / ndarray path ─────────────────────────────────────────────────
    X = df_or_tensor
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)

    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1).cpu().numpy()
    return probs


def predict_single(stock_df: pd.DataFrame, model=None) -> np.ndarray:
    """
    Predict 3-class probabilities for a single stock DataFrame.
    Returns ndarray shape (3,) for a single prediction,
    or (N, 3) if multiple rows are passed and all form a sequence.
    """
    if model is None:
        model = load_lstm()

    device = next(model.parameters()).device
    model.eval()

    feat_cols = model.feature_cols or []
    if feat_cols:
        missing = [c for c in feat_cols if c not in stock_df.columns]
        if missing:
            log.warning(f"predict_single: missing features {missing}, filling with 0")
        X_raw = stock_df.reindex(columns=feat_cols, fill_value=0.0)
    else:
        X_raw = stock_df.select_dtypes(include=[np.number])

    X_raw = X_raw.copy()
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_raw.ffill(inplace=True)
    X_raw.fillna(0, inplace=True)

    X_np = X_raw.values.astype(np.float32)

    if model.scaler is not None:
        try:
            X_np = model.scaler.transform(X_np)
        except Exception as e:
            log.warning(f"Scaler transform failed: {e}; using raw values")

    seq_len = model.seq_len or SEQUENCE_LENGTH
    if len(X_np) < seq_len:
        pad = np.zeros((seq_len - len(X_np), X_np.shape[1]), dtype=np.float32)
        X_np = np.vstack([pad, X_np])

    seq = X_np[-seq_len:]
    X_tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(X_tensor), dim=1).cpu().numpy()[0]
    return probs


if __name__ == "__main__":
    try:
        m = load_lstm()
        print("LSTM model loaded successfully!")
        fc = m.feature_cols or []
        print(f"  Features : {len(fc)}")
        print(f"  seq_len  : {m.seq_len}")
    except Exception as e:
        print(f"Error loading LSTM: {e}")
