# REPLACE ENTIRE FILE: models/xgboost/train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from config.settings import (
    MERGED_CSV, XGB_MODEL_PATH, XGBOOST_PARAMS, XGBOOST_FEATURES,
    TRAIN_END, VAL_START, VAL_END, TEST_START, XGB_RESULTS_PATH,
)

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("xgb_train")

EXT_TO_INT = {-1: 0, 0: 1, 1: 2}
INT_TO_EXT = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
NUM_CLASS   = 3

def encode_labels(y_ext): return np.vectorize(EXT_TO_INT.get)(y_ext)
def decode_labels(y_int): return np.vectorize(INT_TO_EXT.get)(y_int)

def load_data():
    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    return df.sort_values(["Stock", "Date"]).reset_index(drop=True)

def time_split(df):
    return (df[df["Date"] <= TRAIN_END].copy(), df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy(), df[df["Date"] >= TEST_START].copy())

def prepare_xy(df, feature_cols):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_ext = df["label"].astype(int).values
    return X, y_ext, encode_labels(y_ext)

def class_weights_as_sample_weight(y_int, num_class=NUM_CLASS):
    counts  = np.bincount(y_int, minlength=num_class).astype(float)
    weights = len(y_int) / (num_class * np.maximum(counts, 1))
    return weights[y_int]

def print_block(name, y_true_ext, y_pred_ext, y_prob):
    acc = accuracy_score(y_true_ext, y_pred_ext)
    auc = roc_auc_score(y_true_ext, y_prob, multi_class="ovr", labels=[-1, 0, 1], average="macro") if len(np.unique(y_true_ext)) > 1 else float("nan")
    print(f"\n{'='*18} {name} {'='*18}")
    print(f"Accuracy: {acc:.4f}   Macro-OVR AUC: {auc:.4f}")
    print(classification_report(y_true_ext, y_pred_ext, labels=[-1, 0, 1], target_names=CLASS_NAMES, digits=3, zero_division=0))

def main():
    df = load_data()
    train_df, val_df, test_df = time_split(df)
    
    extra = ["hist_vol_20d", "bb_width", "bb_pct", "atr_norm_range", "sector_rel_momentum"]
    feature_cols = [c for c in list(dict.fromkeys(XGBOOST_FEATURES + extra)) if c in train_df.columns]

    X_train, y_train_ext, y_train_int = prepare_xy(train_df, feature_cols)
    X_val,   y_val_ext,   y_val_int   = prepare_xy(val_df,   feature_cols)
    X_test,  y_test_ext,  y_test_int  = prepare_xy(test_df,  feature_cols)

    train_medians = X_train.median(numeric_only=True).to_dict()
    X_train, X_val, X_test = X_train.fillna(train_medians), X_val.fillna(train_medians), X_test.fillna(train_medians)

    sample_weights = class_weights_as_sample_weight(y_train_int)

    xgb_params = {k: v for k, v in XGBOOST_PARAMS.items() if k != "scale_pos_weight"}
    xgb_params.update({"objective": "multi:softprob", "num_class": NUM_CLASS, "eval_metric": "mlogloss"})
    
    # --- NO CALIBRATOR, PURE BASE MODEL TO MAINTAIN WEIGHTS ---
    base = XGBClassifier(**xgb_params, early_stopping_rounds=30)
    base.fit(X_train, y_train_int, sample_weight=sample_weights, eval_set=[(X_val, y_val_int)], verbose=False)

    prob_train, prob_val, prob_test = base.predict_proba(X_train), base.predict_proba(X_val), base.predict_proba(X_test)
    pred_train_ext, pred_val_ext, pred_test_ext = decode_labels(prob_train.argmax(axis=1)), decode_labels(prob_val.argmax(axis=1)), decode_labels(prob_test.argmax(axis=1))

    print_block("TRAIN", y_train_ext, pred_train_ext, prob_train)
    print_block("VALIDATION", y_val_ext, pred_val_ext, prob_val)
    print_block("TEST", y_test_ext, pred_test_ext, prob_test)

    payload = {
        "model": base, # Saving the base model directly
        "base_model": base, 
        "feature_names": list(X_train.columns),
        "train_medians": train_medians,
        "ext_to_int": EXT_TO_INT, "int_to_ext": INT_TO_EXT, "num_class": NUM_CLASS,
        "proba_cols": ["prob_down", "prob_flat", "prob_up"],
    }
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    with open(XGB_MODEL_PATH, "wb") as f: pickle.dump(payload, f)

    res = test_df[["Date", "Stock", "label"]].copy()
    res["Predicted"], res["prob_down"], res["prob_flat"], res["prob_up"] = pred_test_ext, prob_test[:, 0], prob_test[:, 1], prob_test[:, 2]
    res["Confidence"] = prob_test.max(axis=1)
    res.to_csv(XGB_RESULTS_PATH, index=False)

if __name__ == "__main__": main()