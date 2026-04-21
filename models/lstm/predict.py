import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from config.settings   import LSTM_MODEL_PATH, LSTM_SCALER_PATH
from models.lstm.model import LSTMClassifier
from models.lstm.train import _build_sequences


def load_lstm():
    try:
        state  = torch.load(LSTM_MODEL_PATH, map_location="cpu")
        n_feat = state["lstm.weight_ih_l0"].shape[1]
        net    = LSTMClassifier(n_feat)
        net.load_state_dict(state); net.eval()
        with open(LSTM_SCALER_PATH,"rb") as f: scaler = pickle.load(f)
        return {"net":net,"scaler":scaler,"n_features":n_feat}
    except FileNotFoundError:
        raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_PATH}. Run models/lstm/train.py")
    except Exception as e:
        raise RuntimeError(f"LSTM load failed: {e}")


def predict_proba(df, payload=None):
    if payload is None: payload = load_lstm()
    tmp = df.copy()
    if "label" not in tmp.columns: tmp["label"] = 1
    X,_,_ = _build_sequences(tmp, scaler=payload["scaler"], fit_scaler=False)
    if len(X) == 0: return np.full((len(df),3), 1/3)
    with torch.no_grad():
        proba = F.softmax(payload["net"](torch.tensor(X,dtype=torch.float32)),dim=1).numpy()
    pad = np.full((len(df)-len(proba),3), 1/3)
    return np.vstack([pad, proba])