# =============================================================================
# merge_features.py
# Merges technical, fundamental, news, and events CSVs into
# data/merged/merged_final.csv
# =============================================================================

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, EVENTS_CSV, MERGED_CSV,
    TECHNICAL_COLS, EVENTS_COLS, NEWS_CSV
)

def _p(tag, msg):
    print(f"  [{tag}] {msg}")


# ── Load NEWS ────────────────────────────────────────────────────────────────
def _load_news():
    if not os.path.exists(NEWS_CSV):
        return pd.DataFrame()
    return pd.read_csv(NEWS_CSV)


# ── Step 1: Load technical ───────────────────────────────────────────────────
def _load_technical():
    if not os.path.exists(TECHNICAL_CSV):
        _p("✗", "technical.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(TECHNICAL_CSV)

    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)

    df.dropna(subset=["Date", "Close", "Direction"], inplace=True)
    df["Direction"] = df["Direction"].astype(int)

    df.drop_duplicates(subset=["Date", "Stock"], inplace=True)

    return df


# ── Step 2: Lag + rolling features ───────────────────────────────────────────
def _add_lag_rolling(df):
    df = df.copy()
    df.sort_values(["Stock", "Date"], inplace=True)

    lag_map = {
        "Close": [1, 2, 3, 5],
        "RSI": [1],
        "MACD": [1],
        "OBV": [1],
        "Return_1d": [1]
    }

    for col, lags in lag_map.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("Stock")[col].shift(lag)

    for w in [5, 10, 20]:
        df[f"Close_roll_mean_{w}"] = df.groupby("Stock")["Close"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"Close_roll_std_{w}"] = df.groupby("Stock")["Close"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0)
        )

    return df


# ── Step 3: Load fundamental ─────────────────────────────────────────────────
def _load_fundamental():
    if not os.path.exists(FUNDAMENTAL_CSV):
        _p("!", "fundamental.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(FUNDAMENTAL_CSV)

    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)

    # Ensure all columns exist
    cols = [
        "Year", "PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
        "Revenue", "Profit", "Revenue_Growth", "Profit_Growth"
    ]

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    # Clip outliers
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q99)

    return df[[
        "Stock", "Year", "Sector",
        "PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
        "Revenue", "Profit",
        "Revenue_Growth", "Profit_Growth"
    ]]


# ── Step 4: Load events ──────────────────────────────────────────────────────
def _load_events():
    if not os.path.exists(EVENTS_CSV):
        _p("!", "events.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(EVENTS_CSV)

    df.rename(columns={"date": "Date", "symbol": "Stock"}, inplace=True)

    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str)

    df.dropna(subset=["Date", "Stock"], inplace=True)

    for col in ["event_score_max", "event_count", "is_event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["event_category", "event_name"]:
        df[col] = df[col].fillna("NONE")

    return df


# ── MAIN MERGE ───────────────────────────────────────────────────────────────
def merge_all():

    print("\n===== MERGING DATASETS =====")

    tech = _load_technical()
    if tech.empty:
        print("❌ technical.csv missing")
        return

    tech = _add_lag_rolling(tech)

    # Add Year for merge
    tech["Year"] = pd.to_datetime(tech["Date"]).dt.year

    # Load fundamental
    fund = _load_fundamental()

    if not fund.empty:
        merged = tech.merge(fund, on=["Stock", "Year"], how="left")
        print("✅ Fundamental merged:", merged.shape)
    else:
        merged = tech.copy()
        for col in [
            "PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
            "Revenue", "Profit",
            "Revenue_Growth", "Profit_Growth"
        ]:
            merged[col] = np.nan
        merged["Sector"] = "Unknown"

    merged.drop(columns=["Year"], inplace=True, errors="ignore")

    # ── NEWS ─────────────────────────────
    news = _load_news()

    if not news.empty:
        news["Stock"] = news["Stock"].astype(str).str.replace(".NS", "", regex=False)
        news["Date"] = pd.to_datetime(news["Date"]).dt.strftime("%Y-%m-%d")

        cols = ["Date", "Stock", "news_positive", "news_neutral", "news_negative"]
        news = news[cols].groupby(["Date", "Stock"], as_index=False).mean()

        merged = merged.merge(news, on=["Date", "Stock"], how="left")

    for col, val in [
        ("news_positive", 0.333),
        ("news_neutral", 0.334),
        ("news_negative", 0.333)
    ]:
        if col not in merged.columns:
            merged[col] = val
        else:
            merged[col] = merged[col].fillna(val)

    # ── EVENTS ───────────────────────────
    events = _load_events()

    if not events.empty:
        merged = merged.merge(events, on=["Date", "Stock"], how="left")

    for col, val in [
        ("event_score_max", 0.0),
        ("event_count", 0),
        ("is_event", 0)
    ]:
        if col not in merged.columns:
            merged[col] = val
        else:
            merged[col] = merged[col].fillna(val)

    for col in ["event_category", "event_name"]:
        if col not in merged.columns:
            merged[col] = "NONE"
        else:
            merged[col] = merged[col].fillna("NONE")

    # ── FINAL CLEAN ──────────────────────
    merged.dropna(subset=["Close", "Direction", "RSI", "MACD"], inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    merged.sort_values(["Date", "Stock"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Save
    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)

    print("\n✅ Saved:", MERGED_CSV)
    print("Shape:", merged.shape)


if __name__ == "__main__":
    merge_all()