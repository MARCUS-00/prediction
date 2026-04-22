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
    tmp["__orig_idx"] = np.arange(len(tmp))
    if "label" not in tmp.columns: tmp["label"] = 1
    
    out_probas = np.zeros((len(tmp), 2))
    for stock in tmp["Stock"].unique():
        sdf = tmp[tmp["Stock"]==stock].sort_values("Date")
        indices = sdf["__orig_idx"].values
        
        # `_build_sequences` doesn't take `fit_scaler` anymore in our updated file
        try:
            X, _ = _build_sequences(sdf, scaler=payload["scaler"])
        except ValueError:
            X = []
            
        if len(X) > 0:
            with torch.no_grad():
                probas = F.softmax(payload["net"](torch.tensor(X,dtype=torch.float32)),dim=1).numpy()
            pad = np.full((len(sdf)-len(probas), 2), 0.5)
            stock_probas = np.vstack([pad, probas])
        else:
            stock_probas = np.full((len(sdf), 2), 0.5)
            
        out_probas[indices] = stock_probas
        
    return out_probas