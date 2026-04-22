import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config.settings         import (MERGED_CSV, LABEL_MAP, META_MODEL_PATH,
                                      ENSEMBLE_RESULTS_PATH, TRAIN_RATIO, VAL_RATIO,
                                      RANDOM_SEED)
from models.xgboost.predict  import predict_proba as xgb_proba, load_xgb
from models.lstm.predict     import predict_proba  as lstm_proba, load_lstm
from evaluation.metrics      import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _news_proba(df):
    cols = ["news_positive","news_negative"] # Dropped neutral for binary? Or map it?
    if all(c in df.columns for c in cols):
        arr = df[cols].fillna(0.5).values.astype(np.float32)
        return arr / arr.sum(axis=1, keepdims=True).clip(min=1e-9)
    return np.full((len(df),2), 0.5)


def train():
    print("\n" + "="*55)
    print("  ENSEMBLE META-LEARNER — Training")
    print("="*55)

    if not os.path.exists(MERGED_CSV):
        _p("✗", "merged_final.csv not found")
        _p("✗", "Run: python features/merge_features.py"); return {}

    try:
        df = pd.read_csv(MERGED_CSV)
        _p("✓", f"Loaded merged_final.csv  shape={df.shape}")
    except Exception as e:
        _p("✗", f"Cannot load dataset: {e}"); return {}

    df["label"] = df["Direction"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    val_dfs, test_dfs = [], []
    for stock in df["Stock"].unique():
        sdf = df[df["Stock"]==stock].sort_values("Date").reset_index(drop=True)
        n = len(sdf)
        if n == 0: continue
        t = int(n*TRAIN_RATIO); v = int(n*(TRAIN_RATIO+VAL_RATIO))
        val_dfs.append(sdf.iloc[t:v])
        test_dfs.append(sdf.iloc[v:])

    val_df  = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    meta_df = pd.concat([val_df, test_df], ignore_index=True)

    val_size = len(val_df)
    test_size = len(test_df)
    _p("✓", f"Meta stacking: val ({val_size}) + test ({test_size})")

    # Load base models
    try:    xgb = load_xgb()
    except Exception as e: _p("!", f"XGBoost load failed: {e}"); xgb = None

    try:    lstm = load_lstm()
    except Exception as e: _p("!", f"LSTM load failed: {e}"); lstm = None

    parts = []
    for name, fn, payload in [("XGBoost",xgb_proba,xgb),("LSTM",lstm_proba,lstm)]:
        if payload:
            try:
                parts.append(fn(meta_df, payload))
                _p("✓", f"{name} probabilities computed")
            except Exception as e:
                _p("!", f"{name} predict failed: {e} — uniform fill")
                parts.append(np.full((len(meta_df),2), 0.5))
        else:
            parts.append(np.full((len(meta_df),2), 0.5))

    parts.append(_news_proba(meta_df))
    _p("✓", "News probabilities added")

    X_meta = np.hstack(parts)
    y_meta = meta_df["label"].values

    X_val, y_val = X_meta[:val_size], y_meta[:val_size]
    X_test, y_test = X_meta[val_size:], y_meta[val_size:]

    _p("i", "Training Logistic Regression meta-learner on Val data...")
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    meta.fit(X_val, y_val)

    y_pred_train  = meta.predict(X_val)
    y_proba_train = meta.predict_proba(X_val)

    y_pred_val  = meta.predict(X_val)
    y_proba_val = meta.predict_proba(X_val)

    y_pred_test  = meta.predict(X_test)
    y_proba_test = meta.predict_proba(X_test)
    
    # Evaluate: we treat Val as our "Train" for the meta learner
    metrics = evaluate_all(y_val, y_pred_train, y_proba_train,
                           y_val, y_pred_val, y_proba_val,
                           y_test, y_pred_test, y_proba_test, "Ensemble")

    payload = {"meta_model":meta,"metrics":metrics}
    try:
        with open(META_MODEL_PATH,"wb") as f: pickle.dump(payload,f)
        _p("✓", f"Model → {META_MODEL_PATH}")
    except Exception as e:
        _p("✗", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date","Stock","Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
        _p("✓", f"Results → {ENSEMBLE_RESULTS_PATH}")
    except Exception as e:
        _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()