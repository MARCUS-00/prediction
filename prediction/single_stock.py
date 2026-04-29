"""
prediction/single_stock.py
==========================
Single-stock prediction entry point.

BUGS FIXED:
  BUG-4  direction = LABEL_MAP_INV[int(latest["Predicted"])]
         "Predicted" stores internal index {0,1,2}; LABEL_MAP_INV maps
         {0:DOWN, 1:FLAT, 2:UP} — that part was technically OK, BUT
         ensemble/predict.py (old) stored the double-mapped value so the
         index was wrong.  After fixing ensemble/predict.py, this file
         now reads the correct internal index and the lookup is valid.
         Added explicit guard + fallback to Direction_Label column.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd

from config.settings import MERGED_CSV, LABEL_MAP_INV
from models.ensemble.predict import predict_ensemble
from models.xgboost.predict  import load_xgb
try:
    from models.lstm.predict import load_lstm
except Exception:
    load_lstm = None
try:
    from models.ensemble.predict import load_meta
except Exception:
    load_meta = None

from xai.explain_output         import build_bullets, format_output
from prediction.recommendation  import confidence_label, recommendation, expected_movement

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("predict_single")


def predict_single(symbol: str, df: pd.DataFrame = None) -> dict:
    if df is None or df.empty:
        if not os.path.exists(MERGED_CSV):
            return {"error": "merged_final.csv not found. "
                             "Run: python features/merge_features.py"}
        try:
            df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
        except Exception as e:
            return {"error": f"Cannot load dataset: {e}"}

    sdf = df[df["Stock"] == symbol].sort_values("Date").reset_index(drop=True)
    if sdf.empty:
        return {"error": f"No data for '{symbol}'. "
                         "Check symbol is in merged_final.csv"}

    try:
        xgb_p = load_xgb()
    except Exception as e:
        return {"error": f"XGBoost load failed: {e}"}

    lstm_p = None
    if load_lstm is not None:
        try:
            lstm_p = load_lstm()
        except Exception:
            lstm_p = None

    meta_p = None
    if load_meta is not None:
        try:
            meta_p = load_meta()
        except Exception:
            meta_p = None

    try:
        result_df = predict_ensemble(sdf, xgb_p, lstm_p, meta_p)
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    if result_df.empty:
        return {"error": f"No predictions produced for {symbol}"}

    latest = result_df.iloc[-1]

    # ── FIX: prefer explicit Direction_Label; fall back to LABEL_MAP_INV ──────
    if "Direction_Label" in latest.index and latest["Direction_Label"]:
        direction = str(latest["Direction_Label"])
    else:
        pred_int  = int(latest.get("Predicted", 1))
        direction = LABEL_MAP_INV.get(pred_int, "FLAT")

    conf    = float(latest.get("Confidence", 0.0))
    bullets = build_bullets(latest, direction, xgb_p)

    result = {
        "Stock":             symbol,
        "Prediction":        direction,
        "Expected_Movement": expected_movement(latest, direction),
        "Confidence":        f"{conf * 100:.1f}%",
        "Confidence_Level":  confidence_label(conf),
        "Recommendation":    recommendation(direction, conf),
        "Last_Date":         str(latest.get("Date", "N/A")),
        "Last_Close":        f"Rs.{float(latest.get('Close', 0)):.2f}",
        "XAI_Factors":       bullets,
    }
    print(format_output(result))
    return result


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "INFY"
    predict_single(sym)
