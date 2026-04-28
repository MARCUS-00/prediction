# =============================================================================
# app/pages/watchlist_page.py  (FIXED v11)
#
# FIXES vs previous version:
#   - Removed all ensemble and LSTM imports (both DEPRECATED)
#   - Uses XGBoost directly via models/xgboost/predict.py
#   - Removed "neutral (0.33)" placeholder news references from XAI output
#   - Confidence label based on XGBoost calibrated probability
#   - Direction mapped correctly: UP/DOWN (binary, matching alpha_5d target)
#   - Expected movement uses ATR-based estimate, clearly labelled as
#     "volatility estimate" (not a predicted return magnitude)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from config.nifty50_tickers import get_stocks
from config.settings import XGB_MODEL_PATH, MERGED_CSV


# ── Load XGBoost model ──────────────────────────────────────────────────────

@st.cache_resource
def _load_xgb():
    """Load the calibrated XGBoost model. Returns None if not trained yet."""
    if not os.path.exists(XGB_MODEL_PATH):
        return None
    import pickle
    try:
        with open(XGB_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load XGBoost model: {e}")
        return None


@st.cache_data(ttl=3600)
def _load_latest_data():
    """Load the latest rows from merged_final.csv (one row per stock)."""
    if not os.path.exists(MERGED_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(MERGED_CSV)
        df["Date"] = pd.to_datetime(df["Date"])
        # One row per stock: most recent date
        latest = df.sort_values("Date").groupby("Stock").last().reset_index()
        return latest
    except Exception as e:
        st.warning(f"Could not load merged data: {e}")
        return pd.DataFrame()


def _confidence_label(prob):
    if prob >= 0.85:
        return "Strong 🟢"
    elif prob >= 0.70:
        return "Moderate 🟡"
    else:
        return "Weak 🔴"


def _build_xai_factors(row, feat_importances):
    """
    Generate plain-English XAI factors from the top feature values.
    Uses actual feature values from the row — not placeholder text.
    """
    factors = []

    # RSI signal
    if "RSI" in row.index and not pd.isna(row.get("RSI")):
        rsi = float(row["RSI"])
        if rsi > 60:
            factors.append(f"RSI overbought territory ({rsi:.0f}) — momentum signal")
        elif rsi < 40:
            factors.append(f"RSI oversold territory ({rsi:.0f}) — potential reversal")
        else:
            factors.append(f"RSI neutral ({rsi:.0f})")

    # Relative strength vs NIFTY
    if "ret_vs_nifty_5d" in row.index and not pd.isna(row.get("ret_vs_nifty_5d")):
        alpha = float(row["ret_vs_nifty_5d"]) * 100
        if alpha > 1.0:
            factors.append(f"Outperforming NIFTY by +{alpha:.1f}% over 5 days")
        elif alpha < -1.0:
            factors.append(f"Underperforming NIFTY by {alpha:.1f}% over 5 days")

    # Momentum
    if "momentum_5d" in row.index and not pd.isna(row.get("momentum_5d")):
        mom = float(row["momentum_5d"]) * 100
        if mom > 2:
            factors.append(f"Strong 5-day price momentum (+{mom:.1f}%)")
        elif mom < -2:
            factors.append(f"Negative 5-day price momentum ({mom:.1f}%)")

    # Volume spike
    if "vol_spike" in row.index and not pd.isna(row.get("vol_spike")):
        vs = float(row["vol_spike"])
        if vs > 1.5:
            factors.append(f"Elevated volume spike ({vs:.1f}x 10-day average)")

    # News sentiment (only if real, non-zero)
    if "news_score_daily" in row.index:
        ns = float(row.get("news_score_daily", 0.0))
        if abs(ns) > 0.1:
            sentiment = "positive" if ns > 0 else "negative"
            factors.append(f"Recent news sentiment: {sentiment} ({ns:+.2f})")

    # Price position
    if "price_pos_20d" in row.index and not pd.isna(row.get("price_pos_20d")):
        pp = float(row["price_pos_20d"])
        if pp > 0.75:
            factors.append(f"Trading near 20-day high (position: {pp:.0%})")
        elif pp < 0.25:
            factors.append(f"Trading near 20-day low (position: {pp:.0%})")

    if not factors:
        factors = ["Insufficient signal — prediction based on weak features"]

    return factors[:5]


def _get_atr_estimate(stock):
    """Fetch recent ATR for volatility estimate display."""
    try:
        ticker = yf.Ticker(stock + ".NS")
        hist = ticker.history(period="30d", interval="1d")
        if hist.empty or len(hist) < 5:
            return None
        atr = (hist["High"] - hist["Low"]).rolling(14).mean().iloc[-1]
        close = hist["Close"].iloc[-1]
        return float(atr / close * 100)
    except Exception:
        return None


def render_watchlist_page():
    st.header("🏆 Top 10 Stock Watchlist")
    st.caption("Predictions use calibrated XGBoost — alpha-based target (outperformance vs NIFTY)")

    payload = _load_xgb()
    if payload is None:
        st.error(
            "XGBoost model not found. Run the training pipeline first:\n"
            "```bash\n"
            "python features/merge_features.py\n"
            "python models/xgboost/train.py\n"
            "```"
        )
        return

    model          = payload["model"]
    feature_names  = payload["feature_names"]
    train_medians  = payload["train_medians"]
    threshold      = payload.get("threshold", 0.55)

    latest_df = _load_latest_data()
    if latest_df.empty:
        st.error("Could not load merged_final.csv. Run merge_features.py first.")
        return

    stocks = get_stocks()
    available = latest_df[latest_df["Stock"].isin(stocks)].copy()
    if available.empty:
        st.warning("No matching stocks found in merged data.")
        return

    # Build feature matrix for all stocks
    X = pd.DataFrame(index=available.index)
    for col in feature_names:
        if col in available.columns:
            X[col] = available[col]
        else:
            X[col] = train_medians.get(col, 0.0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    X = X.fillna(pd.Series(train_medians))

    proba = model.predict_proba(X)
    available["prob_up"]   = proba[:, 1]
    available["pred_label"] = (proba[:, 1] >= threshold).astype(int)
    available["Direction"]  = available["pred_label"].map({1: "UP", 0: "DOWN"})

    # Rank by highest conviction (distance from 0.5)
    available["conviction"] = (available["prob_up"] - 0.5).abs()
    top10 = available.nlargest(10, "conviction").reset_index(drop=True)

    st.write(f"**Showing top 10 stocks by prediction conviction** "
             f"(as of latest data: {available['Date'].max().date() if 'Date' in available.columns else 'N/A'})")

    # Display cards
    for i, row in top10.iterrows():
        stock     = row["Stock"]
        direction = row["Direction"]
        prob      = float(row["prob_up"]) if direction == "UP" else 1 - float(row["prob_up"])
        conf_lbl  = _confidence_label(prob)
        icon      = "📈" if direction == "UP" else "📉"
        rec       = "BUY" if direction == "UP" else "SELL"

        with st.expander(f"{icon} {stock} — {direction} | Confidence: {prob:.0%} {conf_lbl}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", direction)
                st.metric("Confidence", f"{prob:.1%}")
                st.metric("Recommendation", rec)

            with col2:
                atr_pct = _get_atr_estimate(stock)
                if atr_pct:
                    st.metric("Volatility Estimate (ATR%)", f"±{atr_pct:.1f}%")
                    st.caption("This is an ATR-based volatility estimate, not a predicted return magnitude.")
                if "Close" in row.index and not pd.isna(row.get("Close")):
                    st.metric("Last Close", f"₹{float(row['Close']):.2f}")

            with col3:
                st.write("**Key Factors (XAI):**")
                factors = _build_xai_factors(row, None)
                for f in factors:
                    st.write(f"• {f}")

    st.divider()
    st.caption(
        "⚠️ Predictions are for educational/research purposes only. "
        "Not financial advice. Always perform your own due diligence."
    )