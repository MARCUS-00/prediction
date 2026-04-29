"""
models/ensemble/train_meta.py
==============================
Logistic-Regression meta-learner that combines:
    • XGBoost  – 3 class probabilities  [P(DOWN), P(FLAT), P(UP)]
    • LSTM     – 3 class probabilities  [P(DOWN), P(FLAT), P(UP)]
    • FinBERT  – 3 sentiment scores     [P(pos),  P(neg),  P(neu)]

Total meta-feature vector width: 9 features.

Training protocol (stacking / out-of-fold):
    • Base models were trained on TRAIN split (up to TRAIN_END).
    • Meta-learner trains on VAL split (VAL_START → VAL_END),
      using base-model predictions on data they never trained on.
    • Final evaluation on TEST split (TEST_START → end).

Changes vs. original:
    • FinBERT probabilities ingested as 3 additional inputs.
    • Meta-learner handles 3-class labels {-1, 0, 1}.
    • Per-class metrics and macro OVR AUC reported.
    • Saved payload includes finbert_path for downstream predict.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
)

from config.settings import (
    MERGED_CSV, META_MODEL_PATH, ENSEMBLE_RESULTS_PATH, RANDOM_SEED,
    TRAIN_END, VAL_START, VAL_END, TEST_START,
    NEWS_CSV,
)
from models.xgboost.predict import predict_proba as xgb_proba_fn, load_xgb
from models.lstm.predict    import predict_proba as lstm_proba_fn, load_lstm

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("ensemble_train")

NUM_CLASS   = 3
CLASS_NAMES = ["DOWN", "FLAT", "UP"]

# Path where infer_news.py writes FinBERT scores
FINBERT_SCORES_PATH = os.path.join(os.path.dirname(NEWS_CSV), "finbert_scores.csv")

# ── Ticker normalisation (mirrors merge_features.py) ─────────────────────────
TICKER_ALIAS: dict[str, str] = {
    "M&M": "M&M", "M_M": "M&M", "M-M": "M&M", "MM": "M&M",
    "M&M.NS": "M&M", "M_M.NS": "M&M",
    "BAJAJ-AUTO": "BAJAJ-AUTO", "BAJAJ_AUTO": "BAJAJ-AUTO",
    "BAJAJAUTO": "BAJAJ-AUTO", "BAJAJ AUTO": "BAJAJ-AUTO",
    "BAJAJ-AUTO.NS": "BAJAJ-AUTO",
    "HDFC BANK": "HDFCBANK", "HDFC-BANK": "HDFCBANK",
}


def _norm(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.(NS|BO)$", "", regex=True).str.strip()
    return s.map(lambda t: TICKER_ALIAS.get(t, t))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split(df):
    return (
        df[df["Date"] <= TRAIN_END].copy(),
        df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy(),
        df[df["Date"] >= TEST_START].copy(),
    )


def _load_finbert() -> pd.DataFrame | None:
    """Load FinBERT scores if available; return None and warn if missing."""
    if not os.path.exists(FINBERT_SCORES_PATH):
        log.warning(
            f"FinBERT scores not found at {FINBERT_SCORES_PATH}. "
            "Run models/finbert/infer_news.py first. "
            "Meta-learner will use ZERO FinBERT features as fallback."
        )
        return None
    fb = pd.read_csv(FINBERT_SCORES_PATH, parse_dates=["Date"])
    fb["Stock"] = _norm(fb["Stock"])
    # Aggregate to (Date, Stock) in case of duplicates
    fb = (
        fb.groupby(["Date", "Stock"], as_index=False)
          [["finbert_pos", "finbert_neg", "finbert_neu"]]
          .mean()
    )
    log.info(f"FinBERT scores loaded: {len(fb):,} rows  "
             f"{fb['Stock'].nunique()} tickers")
    return fb


def _get_finbert_features(sdf: pd.DataFrame,
                          fb: pd.DataFrame | None) -> np.ndarray:
    """
    Merge FinBERT scores onto sdf by (Date, Stock).
    Returns ndarray shape (N, 3): [pos, neg, neu].
    Missing rows are filled with [1/3, 1/3, 1/3] (uniform prior).
    """
    n = len(sdf)
    result = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)

    if fb is None or fb.empty:
        return result

    merged = sdf[["Date", "Stock"]].copy().reset_index(drop=True)
    merged["_idx"] = range(n)
    merged = merged.merge(fb, on=["Date", "Stock"], how="left")

    for col_idx, col in enumerate(["finbert_pos", "finbert_neg", "finbert_neu"]):
        if col in merged.columns:
            vals = merged[col].values.astype(float)
            mask = ~np.isnan(vals)
            result[merged.loc[mask, "_idx"].values, col_idx] = vals[mask]

    log.info(f"  FinBERT coverage: "
             f"{int((result[:, 0] != 1/3).sum())}/{n} rows have non-uniform priors")
    return result


def _safe_proba(fn, sdf, payload, name) -> np.ndarray:
    """
    Call a base-model predict_proba function; return (N, 3) array.
    NaN rows indicate the model could not produce a prediction for that row.
    """
    n = len(sdf)
    p = np.full((n, NUM_CLASS), np.nan, dtype=np.float32)
    try:
        r = fn(sdf, payload)
        if r.ndim == 2 and r.shape[1] == NUM_CLASS and r.shape[0] == n:
            p = r.astype(np.float32)
        else:
            log.warning(f"{name} returned shape {r.shape}, expected ({n}, {NUM_CLASS})")
        valid = (~np.isnan(p[:, 0])).sum()
        log.info(f"{name}  valid={valid}/{n}  "
                 f"mean_p_up={np.nanmean(p[:, 2]):.4f}")
    except Exception as e:
        log.warning(f"{name} failed: {e}")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 55)
    print("  ENSEMBLE META-LEARNER  (XGB + LSTM + FinBERT)")
    print("=" * 55)

    if not os.path.exists(MERGED_CSV):
        raise FileNotFoundError(
            f"Missing {MERGED_CSV}. Run features/merge_features.py first.")

    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    df["Stock"] = _norm(df["Stock"])
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    log.info(f"Loaded merged data: {df.shape}")

    _, va_df, te_df = _split(df)
    log.info(f"Val (meta-train)={len(va_df)}  Test={len(te_df)}")
    if len(va_df) == 0 or len(te_df) == 0:
        raise RuntimeError("Empty val/test split — check date ranges in settings.")

    # ── Load base models ──────────────────────────────────────────────────────
    xgb  = load_xgb()
    lstm = load_lstm()
    log.info("Base models (XGBoost + LSTM) loaded")

    # ── FinBERT scores ────────────────────────────────────────────────────────
    fb = _load_finbert()

    # ── Build meta-features for VAL (meta-train) ──────────────────────────────
    log.info("Generating base-model predictions on VAL set ...")
    xv = _safe_proba(xgb_proba_fn,  va_df, xgb,  "XGBoost-val")  # (N, 3)
    lv = _safe_proba(lstm_proba_fn, va_df, lstm, "LSTM-val")      # (N, 3)
    fv = _get_finbert_features(va_df, fb)                          # (N, 3)

    # Mask: rows where both XGB and LSTM returned valid predictions
    mv = (~np.isnan(xv[:, 0])) & (~np.isnan(lv[:, 0]))
    if mv.sum() < 50:
        raise RuntimeError(
            f"Only {int(mv.sum())} rows have valid joint predictions on VAL. "
            "Retrain base models.")

    # Meta-feature matrix: [xgb_down, xgb_flat, xgb_up,
    #                        lstm_down, lstm_flat, lstm_up,
    #                        fb_pos, fb_neg, fb_neu]
    X_meta = np.column_stack([
        xv[mv, 0], xv[mv, 1], xv[mv, 2],
        lv[mv, 0], lv[mv, 1], lv[mv, 2],
        fv[mv, 0], fv[mv, 1], fv[mv, 2],
    ])
    y_meta = va_df["label"].astype(int).values[mv]   # external labels {-1, 0, 1}

    # ── Build meta-features for TEST ──────────────────────────────────────────
    log.info("Generating base-model predictions on TEST set ...")
    xt = _safe_proba(xgb_proba_fn,  te_df, xgb,  "XGBoost-test")
    lt = _safe_proba(lstm_proba_fn, te_df, lstm, "LSTM-test")
    ft = _get_finbert_features(te_df, fb)

    mt = (~np.isnan(xt[:, 0])) & (~np.isnan(lt[:, 0]))
    if mt.sum() == 0:
        raise RuntimeError("No valid joint predictions on TEST set.")

    X_test = np.column_stack([
        xt[mt, 0], xt[mt, 1], xt[mt, 2],
        lt[mt, 0], lt[mt, 1], lt[mt, 2],
        ft[mt, 0], ft[mt, 1], ft[mt, 2],
    ])
    y_test   = te_df["label"].astype(int).values[mt]
    te_valid = te_df.iloc[np.where(mt)[0]].copy().reset_index(drop=True)

    log.info(f"Meta-train shape={X_meta.shape}  test shape={X_test.shape}")

    # ── Train meta-learner ────────────────────────────────────────────────────
    cc = {v: int((y_meta == v).sum()) for v in [-1, 0, 1]}
    log.info(f"Meta-train label counts  DOWN={cc[-1]}  FLAT={cc[0]}  UP={cc[1]}")

    base_lr = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
        C=0.1,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
    )
    min_class_count = min(cc.values())
    if min_class_count >= 3:
        meta = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")
    else:
        log.warning("Too few samples in a class for CV calibration — using raw LR.")
        meta = base_lr

    meta.fit(X_meta, y_meta)
    log.info("Meta-learner trained ✓")

    # ── Evaluation ────────────────────────────────────────────────────────────
    def _eval(name, y_true, X):
        probs = meta.predict_proba(X)
        preds = meta.predict(X)
        acc   = accuracy_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, probs, multi_class="ovr",
                                labels=[-1, 0, 1], average="macro")
        except Exception:
            auc = float("nan")
        print(f"\n{'='*16} {name} {'='*16}")
        print(f"Accuracy: {acc:.4f}   Macro-OVR AUC: {auc:.4f}")
        print(classification_report(
            y_true, preds,
            labels=[-1, 0, 1], target_names=CLASS_NAMES,
            digits=3, zero_division=0,
        ))
        return probs, preds

    print("\n" + "=" * 55)
    print("  ENSEMBLE EVALUATION")
    print("=" * 55)
    pm, _ = _eval("Val (meta-train)", y_meta, X_meta)
    pt, yt_pred = _eval("Test",       y_test, X_test)

    # ── Save meta-learner ─────────────────────────────────────────────────────
    payload = {
        "meta_model":        meta,
        "finbert_available": (fb is not None),
        "finbert_path":      FINBERT_SCORES_PATH,
        # Column layout of X_meta for downstream predict.py
        "meta_feature_cols": [
            "xgb_down", "xgb_flat", "xgb_up",
            "lstm_down", "lstm_flat", "lstm_up",
            "fb_pos",   "fb_neg",   "fb_neu",
        ],
    }
    os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
    with open(META_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"Meta-learner -> {META_MODEL_PATH}")

    # ── Save test results ─────────────────────────────────────────────────────
    res = te_valid[["Date", "Stock"]].copy().reset_index(drop=True)
    res["label"]       = y_test
    res["Predicted"]   = yt_pred
    res["prob_down"]   = pt[:, 0]
    res["prob_flat"]   = pt[:, 1]
    res["prob_up"]     = pt[:, 2]
    res["prob_xgb_up"] = xt[mt, 2]
    res["prob_lstm_up"]= lt[mt, 2]
    res["fb_pos"]      = ft[mt, 0]
    res["Confidence"]  = pt.max(axis=1)
    os.makedirs(os.path.dirname(ENSEMBLE_RESULTS_PATH), exist_ok=True)
    res.to_csv(ENSEMBLE_RESULTS_PATH, index=False)
    log.info(f"Results -> {ENSEMBLE_RESULTS_PATH}")

    return payload


if __name__ == "__main__":
    train()
