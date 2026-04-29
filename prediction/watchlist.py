"""
prediction/watchlist.py
=======================
Generates the daily top-N watchlist using XGBoost predictions.

BUGS FIXED:
  BUG-1  prob_up = proba[:,1]  →  proba[:,1] is P(FLAT), not P(UP).
         Fixed: use proba[:,2] = P(UP).
  BUG-2  pred_label = (proba[:,1] >= threshold) — binary threshold on wrong col.
         Fixed: pred_label = argmax over 3 columns, map to external label.
  BUG-3  Direction = pred_label.map(LABEL_MAP_INV) where pred_label was 0/1,
         so it could only ever produce DOWN or FLAT, never UP.
         Fixed: direction comes from the 3-class argmax result.
  BUG-4  threshold key absent in payload → defaulted silently; now unused.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd

from config.settings           import MERGED_CSV, WATCHLIST_OUTPUT_PATH, LABEL_MAP_INV
from config.nifty50_tickers    import get_stocks
from models.xgboost.predict    import load_xgb, predict_proba as xgb_proba
from xai.explain_output        import build_bullets
from prediction.recommendation import confidence_label, recommendation, expected_movement

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("watchlist")

# Internal index → external label (matches LABEL_MAP_INV convention)
_INT_TO_EXT = {0: -1, 1: 0, 2: 1}   # internal {0,1,2} → external {-1,0,1}
_INT_TO_DIR = {0: "DOWN", 1: "FLAT", 2: "UP"}


def generate_watchlist(df: pd.DataFrame = None, top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        if not os.path.exists(MERGED_CSV):
            log.warning("merged_final.csv not found")
            return pd.DataFrame()
        df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    latest = (df.sort_values("Date")
                .groupby("Stock", as_index=False)
                .last())

    universe = set(get_stocks())
    latest   = latest[latest["Stock"].isin(universe)].reset_index(drop=True)
    if latest.empty:
        log.warning("No matching stocks in merged data.")
        return pd.DataFrame()

    try:
        payload = load_xgb()
    except Exception as e:
        log.error(f"Cannot load XGBoost: {e}")
        return pd.DataFrame()

    try:
        proba = xgb_proba(latest, payload)   # (N, 3): [P(DOWN), P(FLAT), P(UP)]
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        return pd.DataFrame()

    # ── FIX: use correct column index and 3-class argmax ─────────────────────
    prob_up_col   = 2   # index 2 = P(UP)
    int_labels    = proba.argmax(axis=1)                        # 0, 1, or 2
    direction_col = np.vectorize(_INT_TO_DIR.get)(int_labels)  # DOWN/FLAT/UP

    latest["prob_down"]    = proba[:, 0]
    latest["prob_flat"]    = proba[:, 1]
    latest["prob_up"]      = proba[:, prob_up_col]             # FIX: col 2
    latest["pred_int"]     = int_labels
    latest["Direction"]    = direction_col
    # Conviction = distance of P(UP) from 0.5 (how strongly does model favour UP?)
    latest["conviction"]   = (latest["prob_up"] - 1.0 / 3.0).abs()
    latest["Confidence_raw"] = proba.max(axis=1)

    # Rank by UP conviction only — watchlist shows buy candidates
    up_mask = latest["Direction"] == "UP"
    top     = latest[up_mask].nlargest(top_n, "conviction").reset_index(drop=True)
    if top.empty:
        # Fallback: show highest-conviction predictions of any direction
        top = latest.nlargest(top_n, "conviction").reset_index(drop=True)

    rows = []
    for _, r in top.iterrows():
        direction = str(r["Direction"])
        conf      = float(r["Confidence_raw"])
        try:
            bullets = build_bullets(r, direction, payload)
        except Exception:
            bullets = []
        rows.append({
            "Stock":             r["Stock"],
            "Prediction":        direction,
            "Expected_Movement": expected_movement(r, direction),
            "Confidence":        f"{conf * 100:.1f}%",
            "Confidence_Level":  confidence_label(conf),
            "Recommendation":    recommendation(direction, conf),
            "Last_Date":         str(pd.to_datetime(r.get("Date")).date()),
            "Last_Close":        f"Rs.{float(r.get('Close', 0)):.2f}",
            "XAI_Factors":       " | ".join(bullets),
        })

    out = pd.DataFrame(rows)
    try:
        os.makedirs(os.path.dirname(WATCHLIST_OUTPUT_PATH), exist_ok=True)
        out.to_csv(WATCHLIST_OUTPUT_PATH, index=False)
        log.info(f"Watchlist saved → {WATCHLIST_OUTPUT_PATH}")
    except Exception as e:
        log.warning(f"Could not save watchlist CSV: {e}")
    return out


if __name__ == "__main__":
    wl = generate_watchlist()
    print(wl.to_string(index=False) if not wl.empty else "No watchlist produced.")
