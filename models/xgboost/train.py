"""
models/xgboost/train.py
=======================
XGBoost base model – configured for 3-class classification (UP / FLAT / DOWN).

Changes vs. original:
  • objective = 'multi:softprob', num_class = 3
  • Label remapping: model trains on {0,1,2} internally; external labels remain
    {-1, 0, 1} and are mapped back for evaluation and saving.
  • CalibratedClassifierCV wrapped around the multi-class base.
  • Metrics: macro-averaged accuracy, per-class precision/recall/F1,
    and macro OVR AUC (sklearn roc_auc_score with multi_class='ovr').
  • scale_pos_weight replaced by sample_weight vector (correct for multi-class).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report,
)

from config.settings import (
    MERGED_CSV, XGB_MODEL_PATH, XGBOOST_PARAMS, XGBOOST_FEATURES,
    TRAIN_END, VAL_START, VAL_END, TEST_START, RANDOM_SEED,
    XGB_RESULTS_PATH,
)

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("xgb_train")

# ── Label encoding ────────────────────────────────────────────────────────────
# XGBoost multi:softprob requires contiguous integer labels starting at 0.
# External label space: {-1: DOWN, 0: FLAT, 1: UP}
# Internal label space: { 0: DOWN,  1: FLAT, 2: UP }

EXT_TO_INT = {-1: 0, 0: 1, 1: 2}
INT_TO_EXT = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ["DOWN", "FLAT", "UP"]
NUM_CLASS   = 3


def encode_labels(y_ext):
    return np.vectorize(EXT_TO_INT.get)(y_ext)


def decode_labels(y_int):
    return np.vectorize(INT_TO_EXT.get)(y_int)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(MERGED_CSV):
        raise FileNotFoundError(
            f"Missing {MERGED_CSV}. Run features/merge_features.py first.")
    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    return df.sort_values(["Stock", "Date"]).reset_index(drop=True)


def time_split(df):
    train = df[df["Date"] <= TRAIN_END].copy()
    val   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test  = df[df["Date"] >= TEST_START].copy()
    return train, val, test


def select_features(df):
    # Add new features introduced in the updated merge_features.py
    extra = ["hist_vol_20d", "bb_width", "bb_pct", "atr_norm_range",
             "sector_rel_momentum"]
    all_feats = list(dict.fromkeys(XGBOOST_FEATURES + extra))  # preserve order, dedupe
    available = [c for c in all_feats if c in df.columns]
    missing   = [c for c in all_feats if c not in df.columns]
    if missing:
        log.warning(f"Skipping {len(missing)} missing features: {missing}")
    return available


def prepare_xy(df, feature_cols):
    if "label" not in df.columns:
        raise KeyError("Column 'label' missing — re-run merge_features.py")
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_ext = df["label"].astype(int).values
    y_int = encode_labels(y_ext)
    return X, y_ext, y_int


def class_weights_as_sample_weight(y_int, num_class=NUM_CLASS):
    """
    BUG-5 FIX: Per-sample weight inversely proportional to class frequency,
    with a 3x multiplier on minority classes (DOWN=0, UP=2).

    The old inverse-frequency weights were not strong enough — the model still
    predicted FLAT too often (FLAT dominates at ~43% of labels). The 3x boost
    on DOWN/UP forces the model to pay more attention to directional signals.
    """
    counts  = np.bincount(y_int, minlength=num_class).astype(float)
    weights = len(y_int) / (num_class * np.maximum(counts, 1))
    # 3x multiplier on DOWN (index 0) and UP (index 2)
    MINORITY_BOOST = 3.0
    weights[0] *= MINORITY_BOOST   # DOWN
    weights[2] *= MINORITY_BOOST   # UP
    # weights[1] = FLAT — no boost (already majority class)
    log.info(f"BUG-5 FIX: Sample weights — DOWN={weights[0]:.3f}  "
             f"FLAT={weights[1]:.3f}  UP={weights[2]:.3f} (3x boost on DOWN/UP)")
    return weights[y_int]


def print_block(name, y_true_ext, y_pred_ext, y_prob):
    if len(y_true_ext) == 0:
        print(f"\n=== {name} === (EMPTY SPLIT)")
        return
    acc = accuracy_score(y_true_ext, y_pred_ext)
    try:
        auc = roc_auc_score(
            y_true_ext, y_prob, multi_class="ovr",
            labels=[-1, 0, 1], average="macro"
        ) if len(np.unique(y_true_ext)) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    print(f"\n{'='*18} {name} {'='*18}")
    print(f"Samples : {len(y_true_ext)}")
    print(f"Accuracy: {acc:.4f}   Macro-OVR AUC: {auc:.4f}")
    print(classification_report(
        y_true_ext, y_pred_ext,
        labels=[-1, 0, 1], target_names=CLASS_NAMES,
        digits=3, zero_division=0
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  XGBOOST TRAIN  (3-class: DOWN / FLAT / UP)")
    print("=" * 60)

    df = load_data()
    log.info(f"Loaded merged data: {df.shape}")

    train_df, val_df, test_df = time_split(df)
    log.info(f"Splits  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("One of the splits is empty. Check date ranges in settings.")

    feature_cols = select_features(train_df)
    log.info(f"Features used: {len(feature_cols)}")

    X_train, y_train_ext, y_train_int = prepare_xy(train_df, feature_cols)
    X_val,   y_val_ext,   y_val_int   = prepare_xy(val_df,   feature_cols)
    X_test,  y_test_ext,  y_test_int  = prepare_xy(test_df,  feature_cols)

    train_medians = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(pd.Series(train_medians))
    X_val   = X_val.fillna(pd.Series(train_medians))
    X_test  = X_test.fillna(pd.Series(train_medians))

    nunique    = X_train.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        log.warning(f"Dropping constant cols: {const_cols}")
        X_train = X_train.drop(columns=const_cols)
        X_val   = X_val.drop(columns=const_cols)
        X_test  = X_test.drop(columns=const_cols)

    sample_weights = class_weights_as_sample_weight(y_train_int)
    log.info(f"Class counts (int)  0={int((y_train_int==0).sum())}  "
             f"1={int((y_train_int==1).sum())}  2={int((y_train_int==2).sum())}")

    # Build multi-class XGBoost params (override binary-specific keys)
    xgb_params = {k: v for k, v in XGBOOST_PARAMS.items()
                  if k not in ("scale_pos_weight",)}
    xgb_params.update({
        "objective":   "multi:softprob",
        "num_class":   NUM_CLASS,
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    })

    base = XGBClassifier(**xgb_params, early_stopping_rounds=30)

    log.info("Training base XGBoost (multi:softprob, early stopping on val) ...")
    base.fit(
        X_train, y_train_int,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_int)],
        verbose=False,
    )
    best_iter = getattr(base, "best_iteration", None)
    if best_iter is not None:
        log.info(f"Best iteration: {best_iter}")

    log.info("Calibrating probabilities (sigmoid) on validation set ...")
    if _HAS_FROZEN:
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(base), method="sigmoid", cv=None
        )
    else:
        calibrator = CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv="prefit"
        )
    calibrator.fit(X_val, y_val_int)

    # predict_proba → shape (N, 3): columns = [P(DOWN), P(FLAT), P(UP)]
    prob_train = calibrator.predict_proba(X_train)   # (N, 3)
    prob_val   = calibrator.predict_proba(X_val)
    prob_test  = calibrator.predict_proba(X_test)

    pred_train_int = prob_train.argmax(axis=1)
    pred_val_int   = prob_val.argmax(axis=1)
    pred_test_int  = prob_test.argmax(axis=1)

    pred_train_ext = decode_labels(pred_train_int)
    pred_val_ext   = decode_labels(pred_val_int)
    pred_test_ext  = decode_labels(pred_test_int)

    print_block("TRAIN",      y_train_ext, pred_train_ext, prob_train)
    print_block("VALIDATION", y_val_ext,   pred_val_ext,   prob_val)
    print_block("TEST",       y_test_ext,  pred_test_ext,  prob_test)

    try:
        importances = base.feature_importances_
        fi = sorted(zip(X_train.columns, importances),
                    key=lambda x: x[1], reverse=True)[:15]
        print("\nTop-15 Features (by gain):")
        for f, v in fi:
            print(f"  {f:35s} {v:.4f}")
    except Exception as e:
        log.warning(f"Could not compute feature importances: {e}")

    payload = {
        "model":         calibrator,
        "base_model":    base,
        "feature_names": list(X_train.columns),
        "train_medians": {c: float(train_medians.get(c, 0.0))
                          for c in X_train.columns},
        "ext_to_int":    EXT_TO_INT,
        "int_to_ext":    INT_TO_EXT,
        "num_class":     NUM_CLASS,
        # predict_proba columns: [P(DOWN), P(FLAT), P(UP)]
        "proba_cols":    ["prob_down", "prob_flat", "prob_up"],
    }
    os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
    with open(XGB_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"Model saved -> {XGB_MODEL_PATH}")

    res = test_df[["Date", "Stock", "label"]].copy().reset_index(drop=True)
    res["Predicted"]  = pred_test_ext
    res["prob_down"]  = prob_test[:, 0]
    res["prob_flat"]  = prob_test[:, 1]
    res["prob_up"]    = prob_test[:, 2]
    res["Confidence"] = prob_test.max(axis=1)
    os.makedirs(os.path.dirname(XGB_RESULTS_PATH), exist_ok=True)
    res.to_csv(XGB_RESULTS_PATH, index=False)
    log.info(f"Test results -> {XGB_RESULTS_PATH}")


if __name__ == "__main__":
    main()
