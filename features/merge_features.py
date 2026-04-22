# =============================================================================
# merge_features.py  (FIXED + IMPROVED v2)
#
# Changes vs v1:
#   Added cross-sectional features (CS_momentum_rank, CS_rsi_rank, CS_volume_rank)
#   Added pct_from_52w_high, market_vol_20d, intraday_range, gap_pct
#   These are the highest-importance features from ablation experiments.
# =============================================================================

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, EVENTS_CSV, MERGED_CSV,
    TECHNICAL_COLS, EVENTS_COLS, NEWS_CSV, SECTOR_MAP, SECTOR_TO_CODE
)

def _p(tag, msg):
    print(f"  [{tag}] {msg}")


def _add_sector_encoding(df):
    df = df.copy()
    stock_sector = df["Stock"].astype(str).str.replace(".NS", "", regex=False).map(SECTOR_MAP)
    if "Sector" not in df.columns:
        df["Sector"] = stock_sector
    else:
        df["Sector"] = df["Sector"].replace("", np.nan)
        df["Sector"] = df["Sector"].where(df["Sector"].notna(), stock_sector)
        df["Sector"] = df["Sector"].replace("Unknown", np.nan).fillna(stock_sector)
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["sector_encoded"] = df["Sector"].map(SECTOR_TO_CODE).fillna(-1).astype(int)
    return df


# ── Sentiment ─────────────────────────────────────────────────────────────────

def _run_finbert(texts):
    try:
        from transformers import pipeline
        pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                        top_k=None, device=-1)
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
        df["news_positive"] = 0.333; df["news_neutral"] = 0.334; df["news_negative"] = 0.333
        return df
    _p("i", f"Running FinBERT on {len(df)} headlines...")
    pos, neu, neg = _run_finbert(df["News_Text"].fillna("").tolist())
    df = df.copy()
    df["news_positive"] = pos; df["news_neutral"] = neu; df["news_negative"] = neg
    return df


# ── Technical ─────────────────────────────────────────────────────────────────

def _load_technical():
    if not os.path.exists(TECHNICAL_CSV):
        _p("x", "technical.csv not found"); return pd.DataFrame()
    df = pd.read_csv(TECHNICAL_CSV)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    df.dropna(subset=["Date", "Close", "Direction"], inplace=True)
    df["Direction"] = df["Direction"].astype(int)
    df = df[df["Direction"].isin([-1, 1])]
    df.drop_duplicates(subset=["Date", "Stock"], inplace=True)
    return df


def _add_lag_rolling(df):
    df = df.copy()
    df.sort_values(["Stock", "Date"], inplace=True)

    df["EMA_9"] = df.groupby("Stock")["Close"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean())
    df["EMA_50"] = df.groupby("Stock")["Close"].transform(
        lambda x: x.ewm(span=50, adjust=False).mean())

    if all(c in df.columns for c in ["MACD", "MACD_signal"]):
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    if "ATR" in df.columns:
        df["ATR_ratio"] = df["ATR"] / df["Close"].replace(0, np.nan)

    lag_map = {"Close": [1,2,3,5], "RSI": [1], "MACD": [1], "Return_1d": [1]}
    for col, lags in lag_map.items():
        if col not in df.columns: continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("Stock")[col].shift(lag)

    if "RSI" in df.columns:
        df["RSI_change"] = df.groupby("Stock")["RSI"].diff()
    if "Return_1d_lag1" in df.columns:
        df["price_accel"] = df["Return_1d"] - df["Return_1d_lag1"]

    for w in [5, 10, 20]:
        df[f"Close_roll_mean_{w}"] = df.groupby("Stock")["Close"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).mean())
        df[f"Close_roll_std_{w}"]  = df.groupby("Stock")["Close"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).std())

    # Bollinger %B
    bb_mid = df["Close_roll_mean_20"]; bb_std = df["Close_roll_std_20"]
    band_width = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)).replace(0, np.nan)
    df["BB_pct"] = (df["Close"] - (bb_mid - 2*bb_std)) / band_width

    # Volume ratio
    df["Volume_ma20"] = df.groupby("Stock")["Volume"].transform(
        lambda x: x.rolling(20, min_periods=20).mean())
    df["Volume_ratio"] = df["Volume"] / df["Volume_ma20"].replace(0, np.nan)
    df["volume_shock"] = df["Volume_ratio"] - 1.0
    if "OBV" in df.columns:
        df["OBV_change"] = df.groupby("Stock")["OBV"].diff() / df["Volume_ma20"].replace(0, np.nan)

    # Momentum
    df["Momentum_5d"]  = df.groupby("Stock")["Close"].transform(lambda x: x.pct_change(5))
    df["Momentum_10d"] = df.groupby("Stock")["Close"].transform(lambda x: x.pct_change(10))

    if "EMA_20" in df.columns:
        df["EMA_dist"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"].replace(0, np.nan)
    if "EMA_50" in df.columns:
        df["EMA_dist_50"] = (df["Close"] - df["EMA_50"]) / df["EMA_50"].replace(0, np.nan)
    if all(c in df.columns for c in ["EMA_9", "EMA_20"]):
        df["EMA_cross_9_20"] = (df["EMA_9"] > df["EMA_20"]).astype(int)

    if "RSI" in df.columns:
        df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
        df["RSI_oversold"]   = (df["RSI"] < 30).astype(int)

    # ── NEW: Cross-sectional features ─────────────────────────────────────────
    # These rank each stock vs its peers on the same date.
    # They capture relative strength — much more informative than absolute values.
    df["CS_momentum_rank"] = df.groupby("Date")["Momentum_5d"].rank(pct=True)
    df["CS_volume_rank"]   = df.groupby("Date")["Volume_ratio"].rank(pct=True)
    df["CS_rsi_rank"]      = df.groupby("Date")["RSI"].rank(pct=True)
    if "ATR_ratio" in df.columns:
        df["CS_atr_rank"] = df.groupby("Date")["ATR_ratio"].rank(pct=True)

    # ── NEW: 52-week high/low distance ─────────────────────────────────────────
    df["high_252"] = df.groupby("Stock")["Close"].transform(
        lambda x: x.rolling(252, min_periods=50).max())
    df["low_252"] = df.groupby("Stock")["Close"].transform(
        lambda x: x.rolling(252, min_periods=50).min())
    df["pct_from_52w_high"] = (df["Close"] - df["high_252"]) / df["high_252"].replace(0, np.nan)
    df["pct_from_52w_low"] = (df["Close"] - df["low_252"]) / df["low_252"].replace(0, np.nan)

    # ── NEW: Market volatility regime ──────────────────────────────────────────
    daily_mkt = df.groupby("Date")["Return_1d"].mean().rename("_mkt_ret")
    df = df.merge(daily_mkt, on="Date", how="left")
    mkt_vol = daily_mkt.rolling(20).std().rename("market_vol_20d")
    df = df.merge(mkt_vol, on="Date", how="left")
    df.drop(columns=["_mkt_ret"], errors="ignore", inplace=True)

    # ── NEW: Intraday range and overnight gap ──────────────────────────────────
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    prev_close = df.groupby("Stock")["Close"].shift(1)
    df["gap_pct"] = df["Open"] / prev_close.replace(0, np.nan) - 1

    df.drop(columns=["Volume_ma20", "high_252", "low_252"], errors="ignore", inplace=True)
    return df


# ── Fundamental ───────────────────────────────────────────────────────────────

def _load_fundamental():
    if not os.path.exists(FUNDAMENTAL_CSV):
        _p("!", "fundamental.csv not found"); return pd.DataFrame()
    df = pd.read_csv(FUNDAMENTAL_CSV)
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    num_cols = ["PE_Ratio","EPS","ROE","Debt_to_Equity","Revenue","Profit",
                "Revenue_Growth","Profit_Growth"]
    for col in num_cols:
        if col not in df.columns: df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)
    if "Year" not in df.columns: df["Year"] = np.nan
    mapped_sector = df["Stock"].map(SECTOR_MAP)
    if "Sector" not in df.columns:
        df["Sector"] = mapped_sector
    else:
        df["Sector"] = df["Sector"].replace("", np.nan)
        df["Sector"] = df["Sector"].replace("Unknown", np.nan).fillna(mapped_sector)
    df["Sector"] = df["Sector"].fillna("Unknown")
    return df[["Stock","Year","Sector"] + num_cols]


# ── Events ────────────────────────────────────────────────────────────────────

def _load_events():
    if not os.path.exists(EVENTS_CSV):
        _p("!", "events.csv not found"); return pd.DataFrame()
    df = pd.read_csv(EVENTS_CSV)
    df.rename(columns={"date":"Date","symbol":"Stock"}, inplace=True)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str)
    df.dropna(subset=["Date","Stock"], inplace=True)
    for col in ["event_score_max","event_count","is_event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["event_category","event_name"]:
        df[col] = df[col].fillna("NONE")
    return df


# ── MAIN MERGE ────────────────────────────────────────────────────────────────

def merge_all():
    print("\n===== MERGING DATASETS =====")

    tech = _load_technical()
    if tech.empty:
        print("[ERROR] technical.csv missing or empty"); return

    tech = _add_lag_rolling(tech)
    tech["Year"] = pd.to_datetime(tech["Date"]).dt.year

    fund = _load_fundamental()
    if not fund.empty:
        merged = tech.merge(fund, on=["Stock","Year"], how="left")
        fund_num = ["PE_Ratio","EPS","ROE","Debt_to_Equity","Revenue","Profit",
                    "Revenue_Growth","Profit_Growth"]
        merged.sort_values(["Stock","Date"], inplace=True)
        for col in fund_num:
            if col in merged.columns:
                merged[col] = merged.groupby("Stock")[col].transform(lambda x: x.ffill())
        print("[OK] Fundamental merged:", merged.shape)
    else:
        merged = tech.copy()
        for col in ["PE_Ratio","EPS","ROE","Debt_to_Equity","Revenue","Profit",
                    "Revenue_Growth","Profit_Growth"]:
            merged[col] = np.nan
        merged["Sector"] = merged["Stock"].map(SECTOR_MAP).fillna("Unknown")

    merged.drop(columns=["Year"], inplace=True, errors="ignore")
    merged = _add_sector_encoding(merged)

    # News
    news = pd.DataFrame()
    if os.path.exists(NEWS_CSV):
        news = pd.read_csv(NEWS_CSV)
    if not news.empty:
        news["Stock"] = news["Stock"].astype(str).str.replace(".NS","",regex=False)
        news["Date"]  = pd.to_datetime(news["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        news.dropna(subset=["Date","Stock"], inplace=True)
        news = _add_sentiment(news)
        daily_sent = news.groupby(["Date","Stock"], as_index=False).agg(
            news_positive=("news_positive", "mean"),
            news_neutral=("news_neutral", "mean"),
            news_negative=("news_negative", "mean"),
            news_count=("news_positive", "size"),
        )
        daily_sent["news_score"] = daily_sent["news_positive"] - daily_sent["news_negative"]
        daily_sent["has_news"] = 1
        merged = merged.merge(daily_sent, on=["Date","Stock"], how="left")
        _p("OK", f"News merged: {daily_sent.shape}")

    for col, val in [("news_positive",0.333),("news_neutral",0.334),
                     ("news_negative",0.333),("news_score",0.0),
                     ("news_count",0),("has_news",0)]:
        if col not in merged.columns: merged[col] = val
        else: merged[col] = merged[col].fillna(val)
    merged.sort_values(["Stock","Date"], inplace=True)
    merged["news_score_5d"] = merged.groupby("Stock")["news_score"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Events
    events = _load_events()
    if not events.empty:
        agg = {"event_score_max":"max","event_count":"sum","is_event":"max",
               "event_category":"first","event_name":"first"}
        events_agg = events.groupby(["Date","Stock"], as_index=False).agg(agg)
        merged = merged.merge(events_agg, on=["Date","Stock"], how="left")
        _p("OK", f"Events merged: {events_agg.shape}")

    for col, val in [("event_score_max",0.0),("event_count",0),("is_event",0)]:
        if col not in merged.columns: merged[col] = val
        else: merged[col] = merged[col].fillna(val)
    for col in ["event_category","event_name"]:
        if col not in merged.columns: merged[col] = "NONE"
        else: merged[col] = merged[col].fillna("NONE")

    merged.dropna(subset=["Close","Direction"], inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())

    merged.sort_values(["Stock","Date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)

    print("\n[OK] Saved:", MERGED_CSV)
    print("Shape:", merged.shape)
    print("Columns:", len(merged.columns))
    print("\nDirection distribution:\n", merged["Direction"].value_counts())


if __name__ == "__main__":
    merge_all()
