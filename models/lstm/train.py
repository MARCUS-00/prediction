# =============================================================================
# models/lstm/train.py  (FIXED v12)
#
# FIXES vs v9:
#
#   FIX 1: Use date-based split matching project requirement:
#     train < 2024, val = 2024, test = 2025
#     Old code used ratio-based split (70%/15%/15%) which mixes future
#     stocks from 2025 into training — subtle leakage for time-series data.
#
#   FIX 2: news_score_daily is now the column name in merged_final.csv.
#     LSTM_FEATURES in settings.py uses 'news_score' — we alias it here.
#     merge_features.py v12 saves BOTH 'news_score_daily' and 'news_score'
#     so this is now consistent.
#
#   FIX 3: Scaler fit on train_df rows only (was already correct in v9).
#     Kept as-is for clarity.
#
#   All v9 fixes retained:
#     - _build_sequences() has seq_len default (predict.py 3-arg call works)
#     - 0.5% label threshold
#     - Smaller architecture (64 hidden, 1 layer)
#     - Class weighting in CrossEntropyLoss
#     - seq_len saved to pkl
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
                                 LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
                                 LABEL_THRESHOLD, LABEL_HORIZON,
                                 LSTM_MODEL_PATH, LSTM_SCALER_PATH,
                                 RANDOM_SEED)
from models.lstm.model  import LSTMClassifier
from evaluation.metrics import evaluate_all


def _p(tag, msg): print(f"  [{tag}] {msg}", flush=True)

SEQ_LEN = SEQUENCE_LENGTH   # 15 trading days

# FIX 1: Date-based split
TRAIN_END  = "2023-12-31"
VAL_START  = "2024-01-01"
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"


def _engineer_features(df):
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    results = []
    for stock, sdf in df.groupby("Stock", sort=False):
        sdf   = sdf.copy()
        close = sdf["Close"]
        ret1  = sdf["Return_1d"]
        rsi   = sdf["RSI"]
        vol   = sdf["Volume"]

        for lag in [2, 3, 5, 10, 15, 20]:
            sdf[f"ret_{lag}d"] = close.pct_change(lag)

        sdf["hl_ratio"]        = (sdf["High"] - sdf["Low"]) / (close + 1e-8)
        sdf["close_range_pct"] = (close - sdf["Low"]) / (sdf["High"] - sdf["Low"] + 1e-8)
        sdf["gap_close_pct"]   = (sdf["Open"] - close.shift(1)) / (close.shift(1) + 1e-8)
        sdf["rsi_momentum"]    = rsi - rsi.rolling(5).mean()
        sdf["vol10d"]          = ret1.rolling(10).std()
        sdf["sharpe_5d"]       = ret1.rolling(5).mean() / (ret1.rolling(5).std() + 1e-8)
        sdf["norm_mom5"]       = sdf["ret_5d"]  / (sdf["vol10d"] + 1e-8)
        sdf["norm_mom10"]      = sdf["ret_10d"] / (sdf["vol10d"] + 1e-8)
        macd_diff              = sdf["MACD"] - sdf["MACD_signal"]
        sdf["macd_cross"]      = np.sign(macd_diff) - np.sign(macd_diff.shift(1))
        sdf["vol_ratio20"]     = vol / (vol.rolling(20).mean() + 1)

        # Sector alpha
        if "sector_ret_1d" in sdf.columns:
            sdf["return_vs_sector"] = ret1 - sdf["sector_ret_1d"].fillna(0.0)
        else:
            sdf["return_vs_sector"] = 0.0

        # FIX 2: use news_score (aliased from news_score_daily in merge_features v12)
        if "news_score" in sdf.columns:
            sdf["news_rolling_3d"] = sdf["news_score"].fillna(0.0).rolling(3, min_periods=1).mean()
        elif "news_score_daily" in sdf.columns:
            sdf["news_score"] = sdf["news_score_daily"].fillna(0.0)
            sdf["news_rolling_3d"] = sdf["news_score"].rolling(3, min_periods=1).mean()
        else:
            sdf["news_score"]      = 0.0
            sdf["news_rolling_3d"] = 0.0

        results.append(sdf)

    return pd.concat(results, ignore_index=True)


def _add_threshold_labels(df):
    """5-day forward label with 0.5% threshold."""
    df = df.copy().sort_values(["Stock", "Date"]).reset_index(drop=True)
    df["_close_future"] = df.groupby("Stock")["Close"].shift(-LABEL_HORIZON)
    df["_return_fwd"]   = (df["_close_future"] - df["Close"]) / df["Close"].replace(0, np.nan)
    df["label"] = np.where(
        df["_return_fwd"] >  LABEL_THRESHOLD, 1,
        np.where(df["_return_fwd"] < -LABEL_THRESHOLD, 0, np.nan)
    )
    df.drop(columns=["_close_future", "_return_fwd"], inplace=True)
    return df


def _date_split(df):
    """FIX 1: strict date-based split."""
    train_df = df[df["Date"] <= TRAIN_END].copy()
    val_df   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test_df  = df[df["Date"] >= TEST_START].copy()
    return train_df, val_df, test_df


def _build_sequences(stock_df, scaler, feats, seq_len=SEQ_LEN):
    """Build (X, y) sequences. seq_len has default so predict.py 3-arg call works."""
    if len(stock_df) <= seq_len:
        return np.array([]), np.array([])

    df     = stock_df.sort_values("Date").copy().reset_index(drop=True)
    vals   = np.nan_to_num(df[feats].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vals   = scaler.transform(vals)
    labels = df["label"].values

    X, y = [], []
    for i in range(seq_len, len(df)):
        if pd.isna(labels[i]):
            continue
        X.append(vals[i - seq_len : i])
        y.append(int(labels[i]))

    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def _loader(X, y, bs, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)


def train():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

    print("\n" + "=" * 55)
    print("  LSTM — Training (FIXED v12)")
    print("=" * 55)

    if not os.path.exists(MERGED_CSV):
        _p("x", "merged_final.csv not found"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("OK", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("x", f"Cannot load dataset: {e}"); return {}

    df = df.sort_values("Date").reset_index(drop=True)

    _p("i", "Engineering additional features ...")
    df = _engineer_features(df)

    df = _add_threshold_labels(df)
    labeled_count = df["label"].notna().sum()
    _p("OK", f"Labeled rows ({LABEL_HORIZON}d / {LABEL_THRESHOLD*100:.1f}% threshold): "
             f"{labeled_count}/{len(df)}")

    # Resolve features: use LSTM_FEATURES from settings + extras computed above
    extra_feats = ["ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_15d", "ret_20d",
                   "rsi_momentum", "vol10d", "sharpe_5d", "norm_mom5", "norm_mom10",
                   "macd_cross", "vol_ratio20", "return_vs_sector", "news_rolling_3d"]
    feats_wanted = LSTM_FEATURES + [f for f in extra_feats if f not in LSTM_FEATURES]

    for f in feats_wanted:
        if f not in df.columns:
            df[f] = 0.0
    feats_present = [f for f in feats_wanted if f in df.columns]
    _p("i", f"Using {len(feats_present)} features")

    # FIX 1: date-based split
    train_df, val_df, test_df = _date_split(df)
    _p("OK", f"Date split: train:{len(train_df)} ({train_df['Date'].min()}→{TRAIN_END})  "
             f"val:{len(val_df)}  test:{len(test_df)}")

    # Scaler fit on train ONLY (no leakage)
    scaler = RobustScaler()
    train_vals = np.nan_to_num(train_df[feats_present].values.astype(np.float32), nan=0.0)
    scaler.fit(train_vals)

    def _build_split(split_df):
        Xs, ys = [], []
        for stock in split_df["Stock"].unique():
            sdf = split_df[split_df["Stock"] == stock]
            if len(sdf) <= SEQ_LEN: continue
            X, y = _build_sequences(sdf, scaler, feats_present, SEQ_LEN)
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
    total   = len(y_train)
    weights = torch.tensor(total / (2.0 * np.clip(counts, 1, None)), dtype=torch.float32)
    _p("i", f"Class weights: DOWN={weights[0]:.3f}  UP={weights[1]:.3f}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net       = LSTMClassifier(X_train.shape[2]).to(device)
    n_params  = sum(p.numel() for p in net.parameters() if p.requires_grad)
    _p("i", f"Model params: {n_params:,}  (hidden={LSTM_HIDDEN}, layers={LSTM_LAYERS})")

    optimizer = Adam(net.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  patience=max(5, LSTM_PATIENCE // 3), factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)

    best_loss, best_state, patience_cnt = float("inf"), None, 0

    for epoch in range(1, LSTM_EPOCHS + 1):
        net.train()
        for Xb, yb in _loader(X_train, y_train, LSTM_BATCH, shuffle=True):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(net(Xb), yb).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        net.eval(); vl = 0.0; n_batches = 0
        with torch.no_grad():
            for Xb, yb in _loader(X_val, y_val, LSTM_BATCH):
                Xb, yb = Xb.to(device), yb.to(device)
                vl += criterion(net(Xb), yb).item()
                n_batches += 1
        vl /= max(n_batches, 1)
        scheduler.step(vl)

        print(f"    Epoch {epoch:02d}/{LSTM_EPOCHS}  val_loss={vl:.4f}  "
              f"patience={patience_cnt}/{LSTM_PATIENCE}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)

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

    feat_path   = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_features.pkl")
    seqlen_path = LSTM_SCALER_PATH.replace("lstm_scaler.pkl", "lstm_seqlen.pkl")

    try:
        with open(LSTM_SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
        _p("OK", f"Scaler -> {LSTM_SCALER_PATH}")
    except Exception as e: _p("x", f"Scaler save failed: {e}")

    try:
        with open(feat_path, "wb") as f: pickle.dump(feats_present, f)
    except Exception: pass

    try:
        with open(seqlen_path, "wb") as f: pickle.dump(SEQ_LEN, f)
        _p("OK", f"seq_len={SEQ_LEN} saved -> {seqlen_path}")
    except Exception: pass

    return {"net": net, "scaler": scaler, "n_features": X_train.shape[2],
            "feats": feats_present, "seq_len": SEQ_LEN, "metrics": metrics}


if __name__ == "__main__":
    train()