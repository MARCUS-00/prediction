import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import torch
import numpy as np
import pandas as pd

from config.settings import LSTM_MODEL_PATH, DEVICE
from models.lstm.model import LSTMClassifier

log = logging.getLogger("lstm_predict")

def load_lstm():
    """
    Loads the trained PyTorch LSTM model with robust dictionary key extraction.
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {LSTM_MODEL_PATH}. Train it first.")
    
    device = torch.device(DEVICE)
    # Load the checkpoint
    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
    
    # --- ROBUST KEY EXTRACTION ---
    # Claude might have used different names, so we check all common possibilities
    feature_cols = checkpoint.get("feature_cols", 
                   checkpoint.get("features", 
                   checkpoint.get("feature_names", [])))
                   
    scaler = checkpoint.get("scaler", checkpoint.get("scaler_obj", None))
    
    state = checkpoint.get("model_state_dict", 
            checkpoint.get("state_dict", checkpoint))
            
    seq_len = checkpoint.get("seq_len", 
              checkpoint.get("sequence_length", 15))
    
    if not feature_cols:
        log.error(f"Checkpoint keys available: {list(checkpoint.keys())}")
        raise ValueError("Feature columns missing in LSTM checkpoint. See available keys above.")
        
    input_size = len(feature_cols)
    
    # Initialize the model with num_classes=3 to match our trained weights
    net = LSTMClassifier(
        input_size=input_size, 
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2, 
        num_classes=3  
    ).to(device)
    
    net.load_state_dict(state)
    net.eval()
    
    # Attach properties to the model so meta-learner can access them easily
    net.scaler = scaler
    net.feature_cols = feature_cols
    net.seq_len = seq_len
    
    log.info(f"Loaded LSTM from {LSTM_MODEL_PATH} on {device}")
    return net

def predict_proba(X_tensor, model=None):
    """
    Predict probabilities for a tensor of sequences. 
    Used by train_meta.py for ensemble stacking.
    """
    if model is None:
        model = load_lstm()
        
    device = torch.device(DEVICE)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        if isinstance(X_tensor, np.ndarray):
            X_tensor = torch.tensor(X_tensor, dtype=torch.float32)
        X_tensor = X_tensor.to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
    return probs

def predict_single(stock_df):
    """
    Predicts the 3-class probabilities for a single stock dataframe.
    """
    try:
        model = load_lstm()
        
        X_raw = stock_df[model.feature_cols].copy()
        X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_raw.ffill(inplace=True)
        X_raw.fillna(0, inplace=True)
        
        X_scaled = model.scaler.transform(X_raw)
        
        if len(X_scaled) < model.seq_len:
            pad_size = model.seq_len - len(X_scaled)
            pad = np.zeros((pad_size, X_scaled.shape[1]))
            X_scaled = np.vstack([pad, X_scaled])
            
        seq = X_scaled[-model.seq_len:]
        seq_tensor = torch.tensor([seq], dtype=torch.float32)
        
        return predict_proba(seq_tensor, model)[0]
        
    except Exception as e:
        log.error(f"LSTM prediction failed: {e}")
        return np.array([0.0, 1.0, 0.0]) # Default to FLAT on error

if __name__ == "__main__":
    try:
        model = load_lstm()
        print("LSTM model loaded successfully!")
        print(f"Features expected: {len(model.feature_cols)}")
        print(f"Sequence Length: {model.seq_len}")
    except Exception as e:
        print(f"Error loading model: {e}")