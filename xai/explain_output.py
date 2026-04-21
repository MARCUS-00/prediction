import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from xai.feature_importance import importance_for_row


def build_bullets(row: pd.Series, direction: str,
                  xgb_payload=None, max_b=5) -> list:
    bullets  = []
    features = importance_for_row(row, xgb_payload)

    for feat, val, _ in features:
        if len(bullets) >= max_b - 1: break
        if   feat == "RSI":
            tag = "overbought" if val>70 else ("oversold" if val<30 else "neutral zone")
            bullets.append(f"RSI {val:.1f} — {tag}")
        elif feat == "MACD":
            bullets.append(f"MACD {val:.3f} — {'bullish' if val>0 else 'bearish'}")
        elif feat == "EMA_20":
            close = row.get("Close", np.nan)
            if not pd.isna(close):
                bullets.append(f"Price {'above' if close>val else 'below'} EMA-20 ({val:.2f})")
        elif feat == "PE_Ratio":
            bullets.append(f"P/E {val:.1f} — {'expensive' if val>40 else 'fair value'}")
        elif feat == "ATR":
            bullets.append(f"ATR {val:.2f} — {'high' if val>20 else 'low'} volatility")
        elif feat == "is_event" and int(val) == 1:
            bullets.append(f"Event: {row.get('event_name','Market event')}")
        elif feat == "Revenue_Growth":
            bullets.append(f"Revenue growth {val*100:.1f}% — {'strong' if val>0.1 else 'moderate' if val>0 else 'declining'}")
        elif feat == "Profit_Growth":
            bullets.append(f"Profit growth {val*100:.1f}% — {'strong' if val>0.1 else 'moderate' if val>0 else 'declining'}")
        else:
            bullets.append(f"{feat}: {val:.3f} — key driver")

    sl  = str(row.get("news_label","neutral"))
    sscore = float(row.get(f"news_{sl}", 0.333))
    bullets.append(f"News impact: {sl} ({sscore:.2f})")
    return bullets[:max_b]


def format_output(result: dict) -> str:
    lines = ["","="*55, f"  {result.get('Stock','')} — PREDICTION","="*55]
    for k,v in result.items():
        if k in ("XAI_Factors","Stock"): continue
        lines.append(f"  {k:<22}: {v}")
    lines.append("  Key Factors:")
    for b in result.get("XAI_Factors",[]): lines.append(f"    • {b}")
    lines.append("="*55)
    return "\n".join(lines)