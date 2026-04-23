# =============================================================================
# features/merge_features.py  (FIXED v4)
#
# Fixes vs v3:
#   1. Added return_3d, volatility_5d, volume_change — high-signal features
#      requested for accuracy improvement.
#   2. news_score_7d rolling (7-day) added alongside 5d.
#   3. Trend feature: price_above_ema200 (long-term trend regime).
#   4. Removed constant/sparse categorical event columns from final output:
#      event_category and event_name are string columns with very high
#      cardinality and 86% NONE — they add noise, not signal.
#   5. All new features use groupby().shift()/transform() — no future leakage.
# =============================================================================

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, EVENTS_CSV, MERGED_CSV,
    NEWS_CSV, SECTOR_MAP, SECTOR_TO_CODE,
)

def _p(tag, msg):
    print(f"  [{tag}] {msg}")


def _add_sector_encoding(df):
    df = df.copy()
    mapped = df["Stock"].astype(str).str.replace(".NS", "", regex=False).map(SECTOR_MAP)
    if "Sector" not in df.columns:
        df["Sector"] = mapped
    else:
        df["Sector"] = df["Sector"].replace(["", "Unknown"], np.nan)
        df["Sector"] = df["Sector"].where(df["Sector"].notna(), mapped)
    df["Sector"]         = df["Sector"].fillna("Unknown")
    df["sector_encoded"] = df["Sector"].map(SECTOR_TO_CODE).fillna(-1).astype(int)
    return df


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
    g = df.groupby("Stock")

    # ── EMA family ───────────────────────────────────────────────────────────
    df["EMA_9"]   = g["Close"].transform(lambda x: x.ewm(span=9,   adjust=False).mean())
    df["EMA_50"]  = g["Close"].transform(lambda x: x.ewm(span=50,  adjust=False).mean())
    df["EMA_200"] = g["Close"].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    # FIX: Trend feature — is price above 200-day EMA? (long-term regime)
    df["price_above_ema200"] = (df["Close"] > df["EMA_200"]).astype(int)

    # ── MACD histogram ───────────────────────────────────────────────────────
    if all(c in df.columns for c in ["MACD", "MACD_signal"]):
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ── ATR ratio ────────────────────────────────────────────────────────────
    if "ATR" in df.columns:
        df["ATR_ratio"] = df["ATR"] / df["Close"].replace(0, np.nan)

    # ── Lag features ─────────────────────────────────────────────────────────
    for col, lags in {"Close": [1,2,3,5], "RSI": [1], "MACD": [1], "Return_1d": [1]}.items():
        if col not in df.columns: continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

    # ── RSI signals ──────────────────────────────────────────────────────────
    if "RSI" in df.columns:
        df["RSI_change"]     = g["RSI"].transform(lambda x: x.diff())
        df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
        df["RSI_oversold"]   = (df["RSI"] < 30).astype(int)

    # ── Price acceleration ───────────────────────────────────────────────────
    if "Return_1d_lag1" in df.columns:
        df["price_accel"] = df["Return_1d"] - df["Return_1d_lag1"]

    # FIX: return_3d — 3-day lagged return (high signal for momentum)
    df["return_3d"] = g["Close"].transform(lambda x: x.pct_change(3))

    # FIX: volatility_5d — realized volatility over 5 days
    df["volatility_5d"] = g["Return_1d"].transform(
        lambda x: x.rolling(5, min_periods=3).std())

    # FIX: volume_change — day-over-day volume change (surge detection)
    df["volume_change"] = g["Volume"].transform(lambda x: x.pct_change())
    df["volume_change"] = df["volume_change"].clip(-5, 5)  # cap outliers

    # ── Rolling windows ──────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        df[f"Close_roll_mean_{w}"] = g["Close"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).mean())
        df[f"Close_roll_std_{w}"]  = g["Close"].transform(
            lambda x, _w=w: x.rolling(_w, min_periods=_w).std())

    # ── Bollinger %B ─────────────────────────────────────────────────────────
    bb_mid = df["Close_roll_mean_20"]; bb_std = df["Close_roll_std_20"]
    band_w = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)).replace(0, np.nan)
    df["BB_pct"] = (df["Close"] - (bb_mid - 2*bb_std)) / band_w

    # ── Volume features ──────────────────────────────────────────────────────
    df["Volume_ma20"]  = g["Volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["Volume_ratio"] = df["Volume"] / df["Volume_ma20"].replace(0, np.nan)
    df["volume_shock"] = df["Volume_ratio"] - 1.0

    if "OBV" in df.columns:
        df["OBV_change"] = g["OBV"].transform(lambda x: x.diff()) / df["Volume_ma20"].replace(0, np.nan)

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["Momentum_5d"]  = g["Close"].transform(lambda x: x.pct_change(5))
    df["Momentum_10d"] = g["Close"].transform(lambda x: x.pct_change(10))

    # ── EMA distance ─────────────────────────────────────────────────────────
    if "EMA_20" in df.columns:
        df["EMA_dist"]     = (df["Close"] - df["EMA_20"]) / df["EMA_20"].replace(0, np.nan)
    df["EMA_dist_50"]      = (df["Close"] - df["EMA_50"]) / df["EMA_50"].replace(0, np.nan)
    df["EMA_cross_9_20"]   = (df["EMA_9"] > df["EMA_20"]).astype(int)

    # ── Cross-sectional rank features ─────────────────────────────────────────
    df["CS_momentum_rank"] = df.groupby("Date")["Momentum_5d"].rank(pct=True)
    df["CS_volume_rank"]   = df.groupby("Date")["Volume_ratio"].rank(pct=True)
    df["CS_rsi_rank"]      = df.groupby("Date")["RSI"].rank(pct=True)
    if "ATR_ratio" in df.columns:
        df["CS_atr_rank"]  = df.groupby("Date")["ATR_ratio"].rank(pct=True)

    # ── 52-week high/low distance ─────────────────────────────────────────────
    df["high_252"] = g["Close"].transform(lambda x: x.rolling(252, min_periods=50).max())
    df["low_252"]  = g["Close"].transform(lambda x: x.rolling(252, min_periods=50).min())
    df["pct_from_52w_high"] = (df["Close"] - df["high_252"]) / df["high_252"].replace(0, np.nan)
    df["pct_from_52w_low"]  = (df["Close"] - df["low_252"])  / df["low_252"].replace(0, np.nan)

    # ── Market volatility regime ──────────────────────────────────────────────
    daily_mkt = df.groupby("Date")["Return_1d"].mean().rename("_mkt_ret")
    df = df.merge(daily_mkt, on="Date", how="left")
    mkt_vol = daily_mkt.rolling(20).std().rename("market_vol_20d")
    df = df.merge(mkt_vol, on="Date", how="left")
    df.drop(columns=["_mkt_ret"], errors="ignore", inplace=True)

    # ── Intraday range ────────────────────────────────────────────────────────
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    # gap_pct via groupby+shift
    df["prev_close"] = g["Close"].shift(1)
    df["gap_pct"]    = (df["Open"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan)

    # ── Cleanup temp columns ──────────────────────────────────────────────────
    df.drop(columns=["Volume_ma20", "high_252", "low_252", "prev_close", "EMA_200"],
            errors="ignore", inplace=True)
    return df


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
    mapped = df["Stock"].map(SECTOR_MAP)
    if "Sector" not in df.columns:
        df["Sector"] = mapped
    else:
        df["Sector"] = df["Sector"].replace(["", "Unknown"], np.nan)
        df["Sector"] = df["Sector"].where(df["Sector"].notna(), mapped)
    df["Sector"] = df["Sector"].fillna("Unknown")
    return df[["Stock","Year","Sector"] + num_cols]


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
    return df


def merge_all():
    print("\n===== MERGING DATASETS =====")

    tech = _load_technical()
    if tech.empty:
        print("[ERROR] technical.csv missing or empty"); return

    tech = _add_lag_rolling(tech)
    tech["Year"] = pd.to_datetime(tech["Date"]).dt.year

    # ── Fundamental ───────────────────────────────────────────────────────────
    fund = _load_fundamental()
    fund_num = ["PE_Ratio","EPS","ROE","Debt_to_Equity","Revenue","Profit",
                "Revenue_Growth","Profit_Growth"]
    if not fund.empty:
        merged = tech.merge(fund, on=["Stock","Year"], how="left")
        merged.sort_values(["Stock","Date"], inplace=True)
        for col in fund_num:
            if col in merged.columns:
                merged[col] = merged.groupby("Stock")[col].transform(
                    lambda x: x.ffill().bfill())
        _p("OK", f"Fundamental merged: {merged.shape}")
    else:
        merged = tech.copy()
        for col in fund_num: merged[col] = np.nan
        merged["Sector"] = merged["Stock"].map(SECTOR_MAP).fillna("Unknown")

    merged.drop(columns=["Year"], inplace=True, errors="ignore")

    # ── Sector encoding ───────────────────────────────────────────────────────
    merged = _add_sector_encoding(merged)

    # ── News ──────────────────────────────────────────────────────────────────
    news = pd.DataFrame()
    if os.path.exists(NEWS_CSV):
        news = pd.read_csv(NEWS_CSV)
    if not news.empty:
        news["Stock"] = news["Stock"].astype(str).str.replace(".NS","",regex=False)
        news["Date"]  = pd.to_datetime(news["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        news.dropna(subset=["Date","Stock"], inplace=True)
        news = _add_sentiment(news)
        daily_sent = news.groupby(["Date","Stock"], as_index=False).agg(
            news_positive=("news_positive","mean"),
            news_neutral=("news_neutral","mean"),
            news_negative=("news_negative","mean"),
            news_count=("news_positive","size"),
        )
        daily_sent["news_score"] = daily_sent["news_positive"] - daily_sent["news_negative"]
        daily_sent["has_news"]   = 1
        merged = merged.merge(daily_sent, on=["Date","Stock"], how="left")
        _p("OK", f"News merged: {daily_sent.shape}")

    for col, val in [("news_positive",0.333),("news_neutral",0.334),
                     ("news_negative",0.333),("news_score",0.0),
                     ("news_count",0),("has_news",0)]:
        if col not in merged.columns: merged[col] = val
        else: merged[col] = merged[col].fillna(val)

    merged.sort_values(["Stock","Date"], inplace=True)
    merged["news_score_5d"] = merged.groupby("Stock")["news_score"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    # FIX: 7-day rolling news sentiment
    merged["news_score_7d"] = merged.groupby("Stock")["news_score"].transform(
        lambda x: x.rolling(7, min_periods=1).mean())

    # ── Events ────────────────────────────────────────────────────────────────
    events = _load_events()
    if not events.empty:
        # FIX: Only keep numeric event signals — drop sparse string columns
        agg = {"event_score_max":"max","event_count":"sum","is_event":"max"}
        events_agg = events.groupby(["Date","Stock"], as_index=False).agg(agg)
        merged = merged.merge(events_agg, on=["Date","Stock"], how="left")
        _p("OK", f"Events merged: {events_agg.shape}")

    for col, val in [("event_score_max",0.0),("event_count",0),("is_event",0)]:
        if col not in merged.columns: merged[col] = val
        else: merged[col] = merged[col].fillna(val)

    # FIX: Removed event_category and event_name (86% NONE, high cardinality, no signal)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    merged.dropna(subset=["Close","Direction"], inplace=True)
    merged.replace([np.inf,-np.inf], np.nan, inplace=True)

    # FIX: Remove constant columns (zero variance)
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = [c for c in num_cols if merged[c].nunique() <= 1]
    if constant_cols:
        _p("!", f"Dropping constant columns: {constant_cols}")
        merged.drop(columns=constant_cols, inplace=True)
        num_cols = [c for c in num_cols if c not in constant_cols]

    merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median())
    merged.sort_values(["Stock","Date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)

    print(f"\n[OK] Saved: {MERGED_CSV}")
    print(f"Shape: {merged.shape} | Columns: {len(merged.columns)}")
    print("\nDirection distribution:\n", merged["Direction"].value_counts())
    print("\nSector distribution:\n", merged["Sector"].value_counts())


if __name__ == "__main__":
    merge_all()
