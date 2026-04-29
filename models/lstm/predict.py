import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import torch
import pickle
import joblib
import numpy as np
import pandas as pd

# Removed DEVICE import to prevent the ImportError
from config.settings import LSTM_MODEL_PATH
from models.lstm.model import LSTMClassifier

log = logging.getLogger("lstm_predict")

def load_lstm():
    """
    Loads the trained PyTorch LSTM model.
    Automatically finds device and external scaler/feature files.
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {LSTM_MODEL_PATH}. Train it first.")
    
    # 1. Auto-detect device instead of relying on settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
    model_dir = os.path.dirname(LSTM_MODEL_PATH)
    
    # 2. Extract state dict
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    
    # 3. Load Features (Checks dictionary first, then looks for the .pkl file)
    feature_cols = []
    if isinstance(checkpoint, dict) and "feature_cols" in checkpoint:
        feature_cols = checkpoint["feature_cols"]
    else:
        feat_path = os.path.join(model_dir, "lstm_features.pkl")
        if os.path.exists(feat_path):
            with open(feat_path, "rb") as f:
                feature_cols = pickle.load(f)
                
    # 4. Load Scaler (Checks dictionary first, then looks for the .pkl file)
    scaler = None
    if isinstance(checkpoint, dict) and "scaler" in checkpoint:
        scaler = checkpoint["scaler"]
    else:
        scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
    # 5. Sequence Length
    seq_len = 15
    if isinstance(checkpoint, dict) and "seq_len" in checkpoint:
        seq_len = checkpoint["seq_len"]
        
    # 6. Initialize the model
    if not feature_cols:
        log.warning("Could not locate feature names! Defaulting input_size to 25 based on previous logs.")
        input_size = 25
    else:
        input_size = len(feature_cols)
        
    # Initialize with num_classes=3 to match our trained weights
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
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        X_raw = stock_df[model.feature_cols].copy() if model.feature_cols else stock_df.copy()
        X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_raw.ffill(inplace=True)
        X_raw.fillna(0, inplace=True)
        
        if model.scaler:
            X_scaled = model.scaler.transform(X_raw)
        else:
            X_scaled = X_raw.values
        
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
        print(f"Features expected: {len(model.feature_cols) if model.feature_cols else 'Unknown'}")
        print(f"Sequence Length: {model.seq_len}")
    except Exception as e:
        print(f"Error loading model: {e}")