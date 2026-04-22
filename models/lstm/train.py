# =============================================================================
# models/lstm/train.py  (FIXED v3)
#
# Same global-date-split fix as XGBoost train.
# Also fixes _build_sequences signature so lstm/predict.py works correctly.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

from config.settings    import (MERGED_CSV, LSTM_FEATURES, SEQUENCE_LENGTH,
                                 LSTM_EPOCHS, LSTM_BATCH, LSTM_LR, LSTM_PATIENCE,
                                 LABEL_MAP, LSTM_MODEL_PATH, LSTM_SCALER_PATH,
                                 RANDOM_SEED, TRAIN_RATIO, VAL_RATIO)
from models.lstm.model  import LSTMClassifier
from evaluation.metrics import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _global_date_split(df):
    dates = sorted(df["Date"].unique())
    n = len(dates)
    d1 = dates[int(n * TRAIN_RATIO)]
    d2 = dates[int(n * (TRAIN_RATIO + VAL_RATIO))]
    return (df[df["Date"] <  d1].copy(),
            df[(df["Date"] >= d1) & (df["Date"] < d2)].copy(),
            df[df["Date"] >= d2].copy())


def _build_sequences(stock_df, scaler, feats):
    """
    Build LSTM sequences for a single stock's sorted DataFrame.
    scaler and feats are passed explicitly; no implicit globals.
    This signature is also used by lstm/predict.py.
    """
    if not feats or len(stock_df) <= SEQUENCE_LENGTH:
        return np.array([]), np.array([])
    df = stock_df.sort_values("Date").copy()
    vals   = np.nan_to_num(df[feats].values.astype(np.float32), nan=0.0)
    vals   = scaler.transform(vals)
    labels = df["label"].values.astype(int) if "label" in df.columns else np.zeros(len(df), int)
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(vals)):
        X.append(vals[i - SEQUENCE_LENGTH:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


def _loader(X, y, bs, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


def train():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

    print("\n" + "="*55)
    print("  LSTM - Training")
    print("="*55)

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("OK", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)

    feats = [f for f in LSTM_FEATURES if f in df.columns]
    missing = [f for f in LSTM_FEATURES if f not in df.columns]
    if missing: _p("!", f"LSTM features missing (skipped): {missing}")

    # FIXED: global date split
    train_df, val_df, test_df = _global_date_split(df)
    _p("OK", f"Global split: train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    # Fit scaler on training data only
    scaler = RobustScaler()
    train_vals = np.nan_to_num(train_df[feats].values.astype(np.float32), nan=0.0)
    scaler.fit(train_vals)

    # Build sequences per stock within each split
    def _build_split(split_df):
        Xs, ys = [], []
        for stock in split_df["Stock"].unique():
            sdf = split_df[split_df["Stock"] == stock]
            if len(sdf) <= SEQUENCE_LENGTH: continue
            X, y = _build_sequences(sdf, scaler, feats)
            if len(X): Xs.append(X); ys.append(y)
        if not Xs: return np.array([]), np.array([])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = _build_split(train_df)
    X_val,   y_val   = _build_split(val_df)
    X_test,  y_test  = _build_split(test_df)

    if not len(X_train):
        _p("x", "No sequences built."); return {}

    _p("OK", f"Sequences: train:{X_train.shape}  val:{X_val.shape}  test:{X_test.shape}")

    # Class weights
    counts  = np.bincount(y_train, minlength=2)
    weights = torch.tensor(len(y_train) / (2.0 * np.clip(counts, 1, None)), dtype=torch.float32)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net       = LSTMClassifier(X_train.shape[2]).to(device)
    optimizer = Adam(net.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=max(8, LSTM_PATIENCE // 2), factor=0.7
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_loss, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(1, LSTM_EPOCHS + 1):
        net.train()
        for Xb, yb in _loader(X_train, y_train, LSTM_BATCH, shuffle=True):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(net(Xb), yb).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        net.eval(); vl = 0.0; val_batches = 0
        with torch.no_grad():
            for Xb, yb in _loader(X_val, y_val, LSTM_BATCH):
                Xb, yb = Xb.to(device), yb.to(device)
                vl += criterion(net(Xb), yb).item()
                val_batches += 1
        vl /= max(val_batches, 1)
        scheduler.step(vl)
        print(f"    Epoch {epoch:02d}/{LSTM_EPOCHS}  val_loss={vl:.4f}  "
              f"patience={patience_cnt}/{LSTM_PATIENCE}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                _p("i", f"Early stop at epoch {epoch}"); break

    net.load_state_dict(best_state); net.eval(); net = net.cpu()

    def _get_preds(X_data, y_data):
        cpreds, cprobas = [], []
        with torch.no_grad():
            for Xb, _ in _loader(X_data, y_data, LSTM_BATCH):
                lg = net(Xb)
                cpreds.extend(lg.argmax(dim=1).tolist())
                cprobas.extend(torch.softmax(lg, dim=1).numpy())
        return cpreds, np.array(cprobas)

    ptr,  pptr  = _get_preds(X_train, y_train)
    pva,  ppva  = _get_preds(X_val,   y_val)
    preds, probas = _get_preds(X_test, y_test)

    metrics = evaluate_all(y_train, ptr, pptr, y_val, pva, ppva,
                           y_test, preds, probas, "LSTM")

    try:
        torch.save(net.state_dict(), LSTM_MODEL_PATH)
        _p("OK", f"Model -> {LSTM_MODEL_PATH}")
    except Exception as e: _p("x", f"Save failed: {e}")

    try:
        with open(LSTM_SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
        _p("OK", f"Scaler -> {LSTM_SCALER_PATH}")
    except Exception as e: _p("x", f"Scaler save failed: {e}")

    feat_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_features.pkl")
    try:
        with open(feat_path, "wb") as f: pickle.dump(feats, f)
    except Exception: pass

    return {"net": net, "scaler": scaler, "n_features": X_train.shape[2],
            "feats": feats, "metrics": metrics}


if __name__ == "__main__":
    train()
