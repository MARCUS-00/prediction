import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle, random
import numpy as np
import pandas as pd

from config.settings    import (MERGED_CSV, XGBOOST_PARAMS, XGBOOST_FEATURES,
                                 LABEL_MAP, XGB_MODEL_PATH, XGB_RESULTS_PATH,
                                 RANDOM_SEED, TRAIN_RATIO, VAL_RATIO)
from evaluation.metrics import evaluate_all

def _p(tag, msg): print(f"  [{tag}] {msg}")


def _get_X(df):
    avail   = [c for c in XGBOOST_FEATURES if c in df.columns]
    missing = [c for c in XGBOOST_FEATURES if c not in df.columns]
    if missing: _p("!", f"Features not in dataset (skipped): {missing}")
    X = df[avail].copy().apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    return X


def train():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    print("\n" + "="*55)
    print("  XGBOOST — Training")
    print("="*55)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        _p("✗", "xgboost not installed. Run: pip install xgboost"); return {}

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
    train_df,val_df,test_df = df.iloc[:t], df.iloc[t:v], df.iloc[v:]
    _p("✓", f"Split — train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    X_train,y_train = _get_X(train_df), train_df["label"].values
    X_val,  y_val   = _get_X(val_df),   val_df["label"].values
    X_test, y_test  = _get_X(test_df),  test_df["label"].values
    _p("i", f"Training on {X_train.shape[1]} features, {len(X_train)} rows ...")

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=100)

    y_pred_train  = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)
    
    y_pred_val  = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)

    y_pred_test  = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    
    metrics = evaluate_all(y_train, y_pred_train, y_proba_train,
                           y_val, y_pred_val, y_proba_val,
                           y_test, y_pred_test, y_proba_test, "XGBoost")

    payload = {"model":model,"feature_names":X_train.columns.tolist(),"metrics":metrics}
    try:
        with open(XGB_MODEL_PATH,"wb") as f: pickle.dump(payload,f)
        _p("✓", f"Model → {XGB_MODEL_PATH}")
    except Exception as e:
        _p("✗", f"Save failed: {e}")

    try:
        os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
        res = test_df[["Date","Stock","Direction"]].copy()
        res["Predicted"]  = y_pred_test
        res["Confidence"] = y_proba_test.max(axis=1)
        res.to_csv(XGB_RESULTS_PATH, index=False)
        _p("✓", f"Results → {XGB_RESULTS_PATH}")
    except Exception as e:
        _p("!", f"Results save failed: {e}")

    return payload


if __name__ == "__main__":
    train()