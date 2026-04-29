import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from xai.feature_importance import importance_for_row


def _fmt(feat, val, row):
    if feat == "RSI":
        if val > 70:
            tag = "overbought"
        elif val < 30:
            tag = "oversold"
        else:
            tag = "neutral zone"
        return f"RSI {val:.1f} — {tag}"

    if feat == "MACD_hist":
        return f"MACD histogram {val:.3f} — {'bullish' if val > 0 else 'bearish'}"

    if feat == "MACD_signal":
        return f"MACD signal {val:.3f}"

    if feat == "EMA_20":
        close = row.get("Close", np.nan)
        if not pd.isna(close):
            side = "above" if float(close) > val else "below"
            return f"Price {side} EMA-20 ({val:.2f})"
        return f"EMA-20 {val:.2f}"

    if feat == "PE_Ratio":
        return f"P/E {val:.1f} — {'expensive' if val > 40 else 'fair value'}"

    if feat == "ATR":
        return f"ATR {val:.2f} — {'high' if val > 20 else 'low'} volatility"

    if feat == "is_event":
        if int(val) == 1:
            return f"Event flag active: {row.get('event_name', 'market event')}"
        return None

    if feat in ("Revenue_Growth", "Profit_Growth"):
        pretty = "Revenue growth" if feat == "Revenue_Growth" else "Profit growth"
        if val > 0.1:
            tag = "strong"
        elif val > 0:
            tag = "moderate"
        else:
            tag = "declining"
        return f"{pretty} {val * 100:.1f}% — {tag}"

    if feat in ("momentum_5d", "momentum_10d"):
        days = "5-day" if feat == "momentum_5d" else "10-day"
        return f"{days} momentum {val * 100:+.1f}%"

    if feat == "ret_vs_nifty_5d":
        if val > 0.01:
            return f"Outperforming NIFTY by +{val * 100:.1f}% over 5 days"
        if val < -0.01:
            return f"Underperforming NIFTY by {val * 100:.1f}% over 5 days"
        return f"In-line with NIFTY ({val * 100:+.1f}%)"

    if feat == "news_score":
        if abs(val) >= 0.1:
            sentiment = "positive" if val > 0 else "negative"
            return f"Recent news sentiment: {sentiment} ({val:+.2f})"
        return None

    if feat == "price_pos_20d":
        if val > 1.05:
            return f"Trading {(val - 1) * 100:.1f}% above 20-day mean"
        if val < 0.95:
            return f"Trading {(1 - val) * 100:.1f}% below 20-day mean"
        return None

    return f"{feat}: {val:.3f} — key driver"


def build_bullets(row: pd.Series, direction: str,
                  xgb_payload=None, max_b=5) -> list:
    bullets = []
    features = importance_for_row(row, xgb_payload)
    for feat, val, _ in features:
        if len(bullets) >= max_b:
            break
        line = _fmt(feat, val, row)
        if line:
            bullets.append(line)

    if not bullets:
        bullets.append(f"Direction: {direction} (model signals are weak)")
    return bullets[:max_b]


def format_output(result: dict) -> str:
    lines = ["", "=" * 55,
             f"  {result.get('Stock', '')} — PREDICTION", "=" * 55]
    for k, v in result.items():
        if k in ("XAI_Factors", "Stock"):
            continue
        lines.append(f"  {k:<22s}: {v}")
    lines.append("  Key Factors:")
    for b in result.get("XAI_Factors", []):
        lines.append(f"    - {b}")
    lines.append("=" * 55)
    return "\n".join(lines)
