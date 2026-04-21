import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from config.settings    import (MERGED_CSV, LSTM_FEATURES, SEQUENCE_LENGTH,
                                 LSTM_EPOCHS, LSTM_BATCH, LSTM_LR, LSTM_PATIENCE,
                                 LABEL_MAP, LSTM_MODEL_PATH, LSTM_SCALER_PATH,
                                 RANDOM_SEED, TRAIN_RATIO, VAL_RATIO)
from models.lstm.model  import LSTMClassifier
from evaluation.metrics import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _build_sequences(df, scaler):
    feats = [c for c in LSTM_FEATURES if c in df.columns]
    if not feats: return np.array([]), np.array([])
    df     = df.sort_values("Date").copy()
    vals   = np.nan_to_num(df[feats].values.astype(np.float32), nan=0.0)
    vals = scaler.transform(vals)
    labels = df["label"].values.astype(int) if "label" in df.columns else np.zeros(len(df),int)
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(vals)):
        X.append(vals[i-SEQUENCE_LENGTH:i]); y.append(labels[i])
    return np.array(X), np.array(y)


def _loader(X, y, bs, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

    print("\n" + "="*55)
    print("  LSTM — Training")
    print("="*55)

    if not os.path.exists(MERGED_CSV):
        _p("✗", f"merged_final.csv not found")
        _p("✗", "Run: python features/merge_features.py")
        return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("✓", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("✗", f"Cannot load dataset: {e}"); return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    missing_feats = [f for f in LSTM_FEATURES if f not in df.columns]
    if missing_feats:
        _p("!", f"LSTM features missing from dataset: {missing_feats}")

    # Build global scaler on all train portions
    train_dfs = []
    for stock in df["Stock"].unique():
        sdf = df[df["Stock"]==stock].sort_values("Date").reset_index(drop=True)
        n = len(sdf)
        if n == 0: continue
        t = int(n*TRAIN_RATIO)
        train_dfs.append(sdf.iloc[:t])
    
    global_train = pd.concat(train_dfs) if train_dfs else pd.DataFrame()
    scaler = MinMaxScaler()
    existing_feats = [c for c in LSTM_FEATURES if c in global_train.columns]
    if existing_feats and not global_train.empty:
        global_train_vals = np.nan_to_num(global_train[existing_feats].values.astype(np.float32), nan=0.0)
        scaler.fit(global_train_vals)

    Xtr,ytr,Xv,yv,Xte,yte = [],[],[],[],[],[]

    for stock in df["Stock"].unique():
        sdf = df[df["Stock"]==stock].sort_values("Date").reset_index(drop=True)
        n   = len(sdf)
        if n == 0: continue
        t   = int(n*TRAIN_RATIO); v = int(n*(TRAIN_RATIO+VAL_RATIO))
        X1,y1 = _build_sequences(sdf.iloc[:t],   scaler)
        X2,y2 = _build_sequences(sdf.iloc[t:v],  scaler)
        X3,y3 = _build_sequences(sdf.iloc[v:],   scaler)
        if len(X1): Xtr.append(X1); ytr.append(y1)
        if len(X2): Xv.append(X2);  yv.append(y2)
        if len(X3): Xte.append(X3); yte.append(y3)

    if not Xtr:
        _p("✗", "No sequences built. Check SEQUENCE_LENGTH vs data rows.")
        return {}

    X_train,y_train = np.concatenate(Xtr), np.concatenate(ytr)
    X_val,  y_val   = np.concatenate(Xv),  np.concatenate(yv)
    X_test, y_test  = np.concatenate(Xte), np.concatenate(yte)
    _p("✓", f"Sequences — train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")

    net       = LSTMClassifier(X_train.shape[2])
    optimizer = Adam(net.parameters(), lr=LSTM_LR)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state, patience = float("inf"), None, 0

    for epoch in range(1, LSTM_EPOCHS+1):
        net.train()
        for Xb,yb in _loader(X_train,y_train,LSTM_BATCH):
            optimizer.zero_grad(); criterion(net(Xb),yb).backward(); optimizer.step()
        net.eval(); vl = 0.0
        with torch.no_grad():
            for Xb,yb in _loader(X_val,y_val,LSTM_BATCH):
                vl += criterion(net(Xb),yb).item()
        n_batches = max(len(list(_loader(X_val,y_val,LSTM_BATCH))),1)
        vl /= n_batches
        print(f"    Epoch {epoch:02d}/{LSTM_EPOCHS}  val_loss={vl:.4f}  patience={patience}/{LSTM_PATIENCE}")
        if vl < best_loss:
            best_loss=vl; best_state={k:v.clone() for k,v in net.state_dict().items()}; patience=0
        else:
            patience += 1
            if patience >= LSTM_PATIENCE:
                _p("i", f"Early stop at epoch {epoch}"); break

    net.load_state_dict(best_state); net.eval()
    
    def _get_preds(X_data, y_data):
        cpreds, cprobas = [], []
        with torch.no_grad():
            for Xb,_ in _loader(X_data,y_data,LSTM_BATCH):
                lg = net(Xb)
                cpreds.extend(lg.argmax(dim=1).tolist())
                cprobas.extend(torch.softmax(lg,dim=1).numpy())
        return cpreds, np.array(cprobas)

    ptr, pptr = _get_preds(X_train, y_train)
    pva, ppva = _get_preds(X_val, y_val)
    preds, probas = _get_preds(X_test, y_test)
    
    metrics = evaluate_all(y_train, ptr, pptr,
                           y_val, pva, ppva,
                           y_test, preds, probas, "LSTM")

    try:
        torch.save(net.state_dict(), LSTM_MODEL_PATH)
        _p("✓", f"Model → {LSTM_MODEL_PATH}")
    except Exception as e:
        _p("✗", f"Save failed: {e}")

    try:
        with open(LSTM_SCALER_PATH,"wb") as f: pickle.dump(scaler,f)
        _p("✓", f"Scaler → {LSTM_SCALER_PATH}")
    except Exception as e:
        _p("✗", f"Scaler save failed: {e}")

    return {"net":net,"scaler":scaler,"n_features":X_train.shape[2],"metrics":metrics}


if __name__ == "__main__":
    train()