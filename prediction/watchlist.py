import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config.settings           import MERGED_CSV, LABEL_MAP_INV, WATCHLIST_OUTPUT_PATH
from models.ensemble.predict   import predict_ensemble, load_xgb, load_lstm, load_meta
from xai.explain_output        import build_bullets
from prediction.recommendation import confidence_label, recommendation, expected_movement

def _p(tag, msg): print(f"  [{tag}] {msg}")


def generate_watchlist(df: pd.DataFrame = None) -> pd.DataFrame:
    print("\n" + "="*55)
    print("  WATCHLIST - Daily Predictions")
    print("="*55)

    if df is None or df.empty:
        if not os.path.exists(MERGED_CSV):
            _p("x", "merged_final.csv not found. Run: python features/merge_features.py")
            return pd.DataFrame()
        try: df = pd.read_csv(MERGED_CSV)
        except Exception as e:
            _p("x", f"Cannot load dataset: {e}"); return pd.DataFrame()

    try:
        xgb_p=load_xgb(); lstm_p=load_lstm(); meta_p=load_meta()
    except Exception as e:
        _p("x", f"Model load failed: {e}"); return pd.DataFrame()

    stocks  = df["Stock"].unique().tolist()
    records = []
    _p("i", f"Predicting {len(stocks)} stocks ...")

    for sym in stocks:
        try:
            sdf    = df[df["Stock"]==sym].sort_values("Date").reset_index(drop=True)
            if sdf.empty: continue
            res    = predict_ensemble(sdf, xgb_p, lstm_p, meta_p)
            latest = res.iloc[-1]
            direction = LABEL_MAP_INV[int(latest["Predicted"])]
            conf      = float(latest["Confidence"])
            records.append({
                "Stock"            : sym,
                "Prediction"       : direction,
                "Expected_Movement": expected_movement(latest, direction),
                "Confidence"       : f"{conf*100:.1f}%",
                "Confidence_Level" : confidence_label(conf),
                "Recommendation"   : recommendation(direction, conf),
                "Last_Close"       : f"Rs.{float(latest.get('Close',0)):.2f}",
                "Last_Date"        : latest.get("Date", ""),
                "XAI_Factors"      : " | ".join(build_bullets(latest,direction,xgb_p)),
                "_conf"            : conf,
            })
        except Exception as e:
            _p("!", f"{sym} failed: {e}")

    if not records:
        _p("x", "No predictions generated"); return pd.DataFrame()

    wl = pd.DataFrame(records)
    wl.sort_values("_conf", ascending=False, inplace=True)
    wl.drop(columns=["_conf"], inplace=True)
    wl.reset_index(drop=True, inplace=True)

    try:
        os.makedirs(os.path.dirname(WATCHLIST_OUTPUT_PATH), exist_ok=True)
        wl.to_csv(WATCHLIST_OUTPUT_PATH, index=False)
        _p("OK", f"Saved -> {WATCHLIST_OUTPUT_PATH}")
    except Exception as e:
        _p("!", f"Save failed: {e}")

    cols = ["Stock","Prediction","Expected_Movement","Confidence","Recommendation","Confidence_Level"]
    print("\n" + "="*90)
    print("  DAILY WATCHLIST".center(90))
    print("="*90)
    print(wl[[c for c in cols if c in wl.columns]].to_string(index=False))
    print("="*90 + "\n")
    return wl


if __name__ == "__main__":
    generate_watchlist()
