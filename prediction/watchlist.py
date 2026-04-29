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

logging.basicConfig(level=logging.INFO,
                    format="  [%(levelname)s] %(message)s")
log = logging.getLogger("watchlist")


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
    latest = latest[latest["Stock"].isin(universe)].reset_index(drop=True)
    if latest.empty:
        log.warning("No matching stocks in merged data.")
        return pd.DataFrame()

    try:
        payload = load_xgb()
    except Exception as e:
        log.error(f"Cannot load XGBoost: {e}")
        return pd.DataFrame()

    try:
        proba = xgb_proba(latest, payload)
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        return pd.DataFrame()

    threshold = payload.get("threshold", 0.5)
    latest["prob_up"]   = proba[:, 1]
    latest["pred_label"] = (proba[:, 1] >= threshold).astype(int)
    latest["Direction"]  = latest["pred_label"].map(LABEL_MAP_INV)
    latest["Confidence_raw"] = np.maximum(proba[:, 1], 1 - proba[:, 1])

    latest["conviction"] = (latest["prob_up"] - 0.5).abs()
    top = latest.nlargest(top_n, "conviction").reset_index(drop=True)

    rows = []
    for _, r in top.iterrows():
        direction = r["Direction"]
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
        log.info(f"Watchlist saved -> {WATCHLIST_OUTPUT_PATH}")
    except Exception as e:
        log.warning(f"Could not save watchlist CSV: {e}")
    return out


if __name__ == "__main__":
    wl = generate_watchlist()
    print(wl.to_string(index=False) if not wl.empty else "No watchlist produced.")
