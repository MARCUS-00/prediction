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
    cols = ["news_positive","news_neutral","news_negative"]
    if all(c in df.columns for c in cols):
        arr = df[cols].fillna(1/3).values.astype(np.float32)
        return arr / arr.sum(axis=1, keepdims=True).clip(min=1e-9)
    return np.full((len(df),3), 1/3)


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
    df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    t = int(n*TRAIN_RATIO); v = int(n*(TRAIN_RATIO+VAL_RATIO))
    meta_df = pd.concat([df.iloc[t:v], df.iloc[v:]], ignore_index=True)
    _p("✓", f"Meta stacking rows: {len(meta_df)}")

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
                parts.append(np.full((len(meta_df),3), 1/3))
        else:
            parts.append(np.full((len(meta_df),3), 1/3))

    parts.append(_news_proba(meta_df))
    _p("✓", "News probabilities added")

    X_meta = np.hstack(parts)
    y_meta = meta_df["label"].values
    mid    = len(X_meta) // 2

    _p("i", "Training Logistic Regression meta-learner ...")
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    meta.fit(X_meta[:mid], y_meta[:mid])

    y_pred_train  = meta.predict(X_meta[:mid])
    y_proba_train = meta.predict_proba(X_meta[:mid])

    y_pred_test  = meta.predict(X_meta[mid:])
    y_proba_test = meta.predict_proba(X_meta[mid:])
    
    # Meta learner uses half for Train, half for Validate/Test combo
    metrics = evaluate_all(y_meta[:mid], y_pred_train, y_proba_train,
                           y_meta[mid:], y_pred_test, y_proba_test,
                           y_meta[mid:], y_pred_test, y_proba_test, "Ensemble")

    payload = {"meta_model":meta,"metrics":metrics}
    try:
        with open(META_MODEL_PATH,"wb") as f: pickle.dump(payload,f)
        _p("✓", f"Model → {META_MODEL_PATH}")
    except Exception as e:
        _p("✗", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
        res = meta_df.iloc[mid:][["Date","Stock","Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
        _p("✓", f"Results → {ENSEMBLE_RESULTS_PATH}")
    except Exception as e:
        _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()