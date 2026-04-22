# =============================================================================
# merge_features.py  (FIXED + IMPROVED)
# Merges technical, fundamental, news, and events CSVs into
# data/merged/merged_final.csv
#
# Fixes applied:
#   1. KeyError on news_positive/neutral/negative — news.csv has no sentiment
#      columns; score inline via FinBERT (or neutral fallback).
#   2. Rolling/lag features leaked across stock boundaries — now group-aware.
#   3. min_periods=1 on rolling std caused look-ahead on partial windows — fixed
#      to min_periods=w.
#   4. Fundamental forward-fill within stock so yearly data covers daily rows.
#   5. Events duplicate rows on same (Date, Stock) — aggregated before merge.
#   6. dropna(RSI, MACD) dropped valid rows at series start — relaxed to only
#      require Close + Direction.
#   7. Median-impute remaining NaNs instead of silently dropping rows.
#
# Improvements:
#   - BB_pct, Volume_ratio, Momentum_5d/10d, EMA_dist, RSI regime flags.
#   - news_score = positive − negative (single uncorrelated feature).
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


# ── Sentiment helpers ─────────────────────────────────────────────────────────

def _run_finbert(texts):
    try:
        from transformers import pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,
            device=-1,
        )
        results = pipe(texts, batch_size=32, truncation=True, max_length=128)
        pos, neu, neg = [], [], []
        for res_list in results:
            s = {r["label"]: r["score"] for r in res_list}
            pos.append(s.get("positive", 0.333))
            neu.append(s.get("neutral",  0.334))
            neg.append(s.get("negative", 0.333))
        return pos, neu, neg
    except Exception as e:
        _p("!", f"FinBERT unavailable ({e}); using neutral defaults")
        n = len(texts)
        return [0.333]*n, [0.334]*n, [0.333]*n


def _add_sentiment(df):
    if all(c in df.columns for c in ["news_positive", "news_neutral", "news_negative"]):
        return df
    if "News_Text" not in df.columns:
        df["news_positive"] = 0.333
        df["news_neutral"]  = 0.334
        df["news_negative"] = 0.333
        return df
    _p("i", f"Running FinBERT on {len(df)} headlines...")
    pos, neu, neg = _run_finbert(df["News_Text"].fillna("").tolist())
    df = df.copy()
    df["news_positive"] = pos
    df["news_neutral"]  = neu
    df["news_negative"] = neg
    return df


# ── Step 1: Load technical ────────────────────────────────────────────────────

def _load_technical():
    if not os.path.exists(TECHNICAL_CSV):
        _p("x", "technical.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(TECHNICAL_CSV)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    df.dropna(subset=["Date", "Close", "Direction"], inplace=True)
    df["Direction"] = df["Direction"].astype(int)
    df = df[df["Direction"].isin([-1, 1])]
    df.drop_duplicates(subset=["Date", "Stock"], inplace=True)
    return df


# ── Step 2: Lag + rolling + extra features ────────────────────────────────────

def _add_lag_rolling(df):
    df = df.copy()
    df.sort_values(["Stock", "Date"], inplace=True)

    lag_map = {
        "Close":     [1, 2, 3, 5],
        "RSI":       [1],
        "MACD":      [1],
        "OBV":       [1],
        "Return_1d": [1],
    }
    for col, lags in lag_map.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("Stock")[col].shift(lag)

    # Rolling stats with proper min_periods to avoid partial-window leakage
    for w in [5, 10, 20]:
        df[f"Close_roll_mean_{w}"] = df.groupby("Stock")["Close"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).mean()
        )
        df[f"Close_roll_std_{w}"] = df.groupby("Stock")["Close"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).std()
        )

    # Bollinger Band %B
    bb_mid = df["Close_roll_mean_20"]
    bb_std = df["Close_roll_std_20"]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_pct"] = (df["Close"] - bb_lower) / band_width

    # Volume ratio vs 20-day average
    df["Volume_ma20"] = df.groupby("Stock")["Volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    df["Volume_ratio"] = df["Volume"] / df["Volume_ma20"].replace(0, np.nan)

    # Price momentum
    df["Momentum_5d"]  = df.groupby("Stock")["Close"].transform(lambda x: x.pct_change(5))
    df["Momentum_10d"] = df.groupby("Stock")["Close"].transform(lambda x: x.pct_change(10))

    # EMA distance (normalised)
    if "EMA_20" in df.columns:
        df["EMA_dist"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"].replace(0, np.nan)

    # RSI regime flags
    if "RSI" in df.columns:
        df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
        df["RSI_oversold"]   = (df["RSI"] < 30).astype(int)

    df.drop(columns=["Volume_ma20"], errors="ignore", inplace=True)
    return df


# ── Step 3: Load fundamental ──────────────────────────────────────────────────

def _load_fundamental():
    if not os.path.exists(FUNDAMENTAL_CSV):
        _p("!", "fundamental.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(FUNDAMENTAL_CSV)
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    num_cols = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                "Revenue", "Profit", "Revenue_Growth", "Profit_Growth"]
    for col in num_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)
    if "Year" not in df.columns:
        df["Year"] = np.nan
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"
    return df[["Stock", "Year", "Sector"] + num_cols]


# ── Step 4: Load events ───────────────────────────────────────────────────────

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


# ── MAIN MERGE ────────────────────────────────────────────────────────────────

def merge_all():
    print("\n===== MERGING DATASETS =====")

    tech = _load_technical()
    if tech.empty:
        print("❌ technical.csv missing or empty")
        return

    tech = _add_lag_rolling(tech)
    tech["Year"] = pd.to_datetime(tech["Date"]).dt.year

    # ── Fundamental ──────────────────────────────────────────────────────────
    fund = _load_fundamental()
    if not fund.empty:
        merged = tech.merge(fund, on=["Stock", "Year"], how="left")
        fund_num_cols = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                         "Revenue", "Profit", "Revenue_Growth", "Profit_Growth"]
        merged.sort_values(["Stock", "Date"], inplace=True)
        for col in fund_num_cols:
            if col in merged.columns:
                merged[col] = merged.groupby("Stock")[col].transform(
                    lambda x: x.ffill().bfill()
                )
        print("✅ Fundamental merged:", merged.shape)
    else:
        merged = tech.copy()
        for col in ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                    "Revenue", "Profit", "Revenue_Growth", "Profit_Growth"]:
            merged[col] = np.nan
        merged["Sector"] = "Unknown"

    merged.drop(columns=["Year"], inplace=True, errors="ignore")

    # ── News ─────────────────────────────────────────────────────────────────
    news = pd.DataFrame()
    if os.path.exists(NEWS_CSV):
        news = pd.read_csv(NEWS_CSV)

    if not news.empty:
        news["Stock"] = news["Stock"].astype(str).str.replace(".NS", "", regex=False)
        news["Date"]  = pd.to_datetime(news["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        news.dropna(subset=["Date", "Stock"], inplace=True)
        news = _add_sentiment(news)
        sent_cols = ["Date", "Stock", "news_positive", "news_neutral", "news_negative"]
        daily_sent = news[sent_cols].groupby(["Date", "Stock"], as_index=False).mean()
        daily_sent["news_score"] = daily_sent["news_positive"] - daily_sent["news_negative"]
        merged = merged.merge(daily_sent, on=["Date", "Stock"], how="left")
        _p("✓", f"News merged: {daily_sent.shape}")

    for col, val in [
        ("news_positive", 0.333),
        ("news_neutral",  0.334),
        ("news_negative", 0.333),
        ("news_score",    0.0),
    ]:
        if col not in merged.columns:
            merged[col] = val
        else:
            merged[col] = merged[col].fillna(val)

    # ── Events ───────────────────────────────────────────────────────────────
    events = _load_events()
    if not events.empty:
        agg = {
            "event_score_max": "max",
            "event_count":     "sum",
            "is_event":        "max",
            "event_category":  "first",
            "event_name":      "first",
        }
        events_agg = events.groupby(["Date", "Stock"], as_index=False).agg(agg)
        merged = merged.merge(events_agg, on=["Date", "Stock"], how="left")
        _p("✓", f"Events merged: {events_agg.shape}")

    for col, val in [
        ("event_score_max", 0.0),
        ("event_count",     0),
        ("is_event",        0),
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

    # ── Final clean ───────────────────────────────────────────────────────────
    merged.dropna(subset=["Close", "Direction"], inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())

    merged.sort_values(["Date", "Stock"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)

    print("\n✅ Saved:", MERGED_CSV)
    print("Shape:", merged.shape)
    print("\nDirection distribution:\n", merged["Direction"].value_counts())


if __name__ == "__main__":
    merge_all()