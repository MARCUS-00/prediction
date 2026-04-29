# REPLACE ENTIRE FILE: features/merge_features.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, NEWS_CSV, EVENTS_CSV, MERGED_CSV,
    LABEL_HORIZON, DATE_START, DATE_END,
    SECTOR_MAP, SECTOR_TO_CODE,
)

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("merge_features")

TICKER_ALIAS: dict[str, str] = {
    "M&M": "M&M", "M_M": "M&M", "M-M": "M&M", "MM": "M&M", "M&M.NS": "M&M", "M_M.NS": "M&M",
    "BAJAJ-AUTO": "BAJAJ-AUTO", "BAJAJ_AUTO": "BAJAJ-AUTO", "BAJAJAUTO": "BAJAJ-AUTO", "BAJAJ AUTO": "BAJAJ-AUTO", "BAJAJ-AUTO.NS": "BAJAJ-AUTO",
    "HDFC BANK": "HDFCBANK", "HDFC-BANK": "HDFCBANK",
}

# --- OPTION B: IMPROVED 3-CLASS THRESHOLDS ---
UP_THRESH   =  0.02   # > +2.0% -> UP
DOWN_THRESH = -0.02   # < -2.0% -> DOWN

def normalise_ticker(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.(NS|BO)$", "", regex=True).str.strip()
    return s.map(lambda t: TICKER_ALIAS.get(t, t))

def _remove_constant_cols(df, feature_cols):
    drop = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if drop:
        log.warning(f"Dropping constant cols: {drop}")
    return [c for c in feature_cols if c not in drop]

_NIFTY_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "merged", "_nifty_cache.csv")

def _load_nifty():
    raw = pd.DataFrame()
    try:
        raw = yf.download("^NSEI", start=DATE_START, end=DATE_END, auto_adjust=True, progress=False)
    except Exception as e:
        log.warning(f"yfinance fetch failed: {e}")
    if raw is None or raw.empty:
        if os.path.exists(_NIFTY_CACHE):
            return pd.read_csv(_NIFTY_CACHE, parse_dates=["Date"])
        raise RuntimeError("Empty data for ^NSEI")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    nifty = raw[["Close"]].copy()
    nifty.index = pd.to_datetime(nifty.index)
    nifty.index.name = "Date"
    nifty = nifty.sort_index()
    nifty["nifty_close"]  = nifty["Close"]
    nifty["nifty_ret_1d"] = nifty["Close"].pct_change(1).shift(1)
    nifty["nifty_ret_5d"] = nifty["Close"].pct_change(5).shift(1)
    nifty = nifty.drop(columns=["Close"]).reset_index()
    try:
        os.makedirs(os.path.dirname(_NIFTY_CACHE), exist_ok=True)
        nifty.to_csv(_NIFTY_CACHE, index=False)
    except Exception: pass
    return nifty

def merge_all():
    print("\n" + "=" * 60)
    print("  MERGE FEATURES  (Dynamic Labels & Fixed News)")
    print("=" * 60)

    tech = pd.read_csv(TECHNICAL_CSV)
    tech["Date"]  = pd.to_datetime(tech["Date"])
    tech["Stock"] = normalise_ticker(tech["Stock"])
    tech = tech[(tech["Date"] >= DATE_START) & (tech["Date"] <= DATE_END)]
    tech = tech.sort_values(["Stock", "Date"]).drop_duplicates(subset=["Stock", "Date"], keep="last")
    
    g = tech.groupby("Stock", group_keys=False)
    tech["momentum_5d"]       = g["Close"].transform(lambda x: x.pct_change(5))
    tech["momentum_10d"]      = g["Close"].transform(lambda x: x.pct_change(10))
    tech["momentum_diff"]     = tech["momentum_5d"] - tech["momentum_10d"]
    tech["momentum_strength"] = tech.groupby("Stock")["momentum_5d"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    tech["volatility_ratio"]   = tech["ATR"] / (tech["Close"].abs() + 1e-8)
    
    tech["_daily_ret"]  = g["Close"].transform(lambda x: x.pct_change(1))
    tech["hist_vol_20d"] = tech.groupby("Stock")["_daily_ret"].transform(lambda x: x.rolling(20, min_periods=5).std())
    tech.drop(columns=["_daily_ret"], inplace=True)

    bb_mid = g["Close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    bb_std = g["Close"].transform(lambda x: x.rolling(20, min_periods=5).std())
    tech["bb_upper"], tech["bb_lower"] = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std
    tech["bb_width"] = (tech["bb_upper"] - tech["bb_lower"]) / (bb_mid.abs() + 1e-8)
    tech["bb_pct"]   = (tech["Close"] - tech["bb_lower"]) / (tech["bb_upper"] - tech["bb_lower"] + 1e-8)
    tech.drop(columns=["bb_upper", "bb_lower"], inplace=True)
    tech["atr_norm_range"] = (tech["High"] - tech["Low"]) / (tech["ATR"] + 1e-8)

    vol_ma20    = g["Volume"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    close_max20 = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).max())
    close_ma20  = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    tech["vol_spike"]     = (tech["Volume"] > vol_ma20 * 1.5).astype(int)
    tech["vol_breakout"]  = (tech["Close"]  > close_max20.shift(1)).astype(int)
    tech["price_pos_20d"] = tech["Close"] / (close_ma20 + 1e-8)

    ema12, ema26 = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean()), g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    if "MACD_signal" not in tech.columns or tech["MACD_signal"].isna().all():
        tech["MACD_signal"] = macd.groupby(tech["Stock"]).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    tech["MACD_hist"] = macd - tech["MACD_signal"]
    if "EMA_20" not in tech.columns or tech["EMA_20"].isna().all():
        tech["EMA_20"] = g["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())

    nifty = _load_nifty()
    tech = tech.merge(nifty[["Date", "nifty_ret_1d", "nifty_ret_5d"]], on="Date", how="left")
    tech["ret_vs_nifty_1d"] = tech["Return_1d"]   - tech["nifty_ret_1d"]
    tech["ret_vs_nifty_5d"] = tech["momentum_5d"] - tech["nifty_ret_5d"]

    tech["Sector"] = tech["Stock"].map(SECTOR_MAP).fillna("Unknown")
    tech["sector_encoded"] = tech["Sector"].map(SECTOR_TO_CODE).fillna(-1).astype(int)
    sector_daily = tech.groupby(["Date", "Sector"])["Return_1d"].mean().rename("sector_ret_1d_raw").reset_index()
    sector_daily = sector_daily.sort_values(["Sector", "Date"])
    sector_daily["sector_ret_1d"] = sector_daily.groupby("Sector")["sector_ret_1d_raw"].transform(lambda x: x.shift(1))
    sector_daily["sector_ret_5d"] = sector_daily.groupby("Sector")["sector_ret_1d_raw"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    sector_daily = sector_daily.drop(columns=["sector_ret_1d_raw"])
    tech = tech.merge(sector_daily, on=["Date", "Sector"], how="left")
    tech["sector_ret_1d"] = tech["sector_ret_1d"].fillna(0.0)
    tech["sector_ret_5d"] = tech["sector_ret_5d"].fillna(0.0)
    tech["return_vs_sector"] = tech["Return_1d"].fillna(0.0) - tech["sector_ret_1d"]
    tech["sector_rel_momentum"] = tech["momentum_10d"] - tech["sector_ret_5d"]

    if os.path.exists(FUNDAMENTAL_CSV):
        fund = pd.read_csv(FUNDAMENTAL_CSV)
        fund["Stock"] = normalise_ticker(fund["Stock"])
        fund["Date"]  = pd.to_datetime(fund["Year"].astype(int).astype(str) + "-12-31")
        fund = fund.sort_values(["Stock", "Date"]).drop_duplicates(subset=["Stock", "Date"], keep="last")
        df = pd.merge_asof(tech.sort_values("Date"), fund[["Stock", "Date", "PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]].sort_values("Date"), on="Date", by="Stock", direction="backward")
    else:
        df = tech.copy()
        for c in ["PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]: df[c] = np.nan

    # --- FIXED NEWS PIPELINE ---
    if os.path.exists(NEWS_CSV):
        news = pd.read_csv(NEWS_CSV)
        news["Date"]  = pd.to_datetime(news["Date"])
        news["Stock"] = normalise_ticker(news["Stock"])
        if "news_score" not in news.columns: news["news_score"] = 0.0
        news = news.groupby(["Date", "Stock"], as_index=False)["news_score"].mean()
        df = df.merge(news, on=["Date", "Stock"], how="left")
        
        # Forward fill news to propagate sentiment over weekends/quiet days before defaulting to 0
        df["news_score"] = df.groupby("Stock")["news_score"].fillna(method='ffill', limit=3)
        df["news_score"] = df["news_score"].fillna(0.0)
    else:
        df["news_score"] = 0.0

    df["has_news"] = (df["news_score"] != 0).astype(int)
    df = df.sort_values(["Stock", "Date"])
    df["news_rolling_3d"] = df.groupby("Stock")["news_score"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["news_decay"] = df.groupby("Stock")["news_score"].transform(lambda x: x.ewm(span=5, adjust=False).mean())

    if os.path.exists(EVENTS_CSV):
        events = pd.read_csv(EVENTS_CSV)
        events = events.rename(columns={"date": "Date", "symbol": "Stock"})
        events["Date"]  = pd.to_datetime(events["Date"])
        events["Stock"] = normalise_ticker(events["Stock"])
        events_agg = events.groupby(["Date", "Stock"], as_index=False).agg(event_score_max=("event_score_max", "max"), is_event=("is_event", "max"))
        df = df.merge(events_agg, on=["Date", "Stock"], how="left")
    else:
        df["event_score_max"], df["is_event"] = 0.0, 0

    df["is_event"] = df["is_event"].fillna(0).astype(int)
    df["event_score_max"] = df["event_score_max"].fillna(0.0)
    df["event_strength"]  = np.log1p(df["event_score_max"].clip(lower=0))
    df["event_impact_decay"] = df.groupby("Stock")["event_strength"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["alpha_strength"] = df.groupby("Stock")["Close"].transform(lambda x: x.pct_change(5).abs())

    # --- FIXED LABEL GENERATION ---
    df["_fwd_close"] = df.groupby("Stock")["Close"].shift(-LABEL_HORIZON)
    df["_fwd_ret"]   = (df["_fwd_close"] - df["Close"]) / df["Close"].replace(0, np.nan)
    df["label"] = np.where(df["_fwd_ret"] > UP_THRESH, 1, np.where(df["_fwd_ret"] < DOWN_THRESH, -1, 0))
    df.loc[df["_fwd_ret"].isna(), "label"] = np.nan
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    valid, up, flat, down = len(df), int((df["label"]==1).sum()), int((df["label"]==0).sum()), int((df["label"]==-1).sum())
    log.info(f"Labels: total={valid} UP={up} ({up/valid*100:.1f}%) FLAT={flat} ({flat/valid*100:.1f}%) DOWN={down} ({down/valid*100:.1f}%)")

    drop_cols = ["_fwd_close", "_fwd_ret", "Direction", "Year", "Sector"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0)
    df[str_cols] = df[str_cols].fillna("")

    feature_candidates = [c for c in df.columns if c not in ["Date", "Stock", "label", "Open", "High", "Low", "Close", "Volume"]]
    feature_candidates = _remove_constant_cols(df, feature_candidates)

    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)
    log.info(f"Saved: {MERGED_CSV}")

if __name__ == "__main__":
    merge_all()