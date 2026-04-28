# =============================================================================
# features/merge_features.py  (FIXED v13)
#
# FIXES vs v12:
#   FIX 1: Label — stronger alpha threshold (|alpha_5d| > 0.015 not 0.01)
#   FIX 2: Sentiment — per-stock z-score normalization (removes +0.19 bias)
#   FIX 3: Feature list pruned to strong set only (FEATURES constant exported)
#   FIX 4: Added momentum_strength and volatility_ratio features
#   FIX 5: M&M validation output at end
# =============================================================================

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, NEWS_CSV, EVENTS_CSV, MERGED_CSV,
    SECTOR_MAP,
)

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

# ---------------------------------------------------------------------------
# CANONICAL FEATURE LIST  (imported by train scripts)
# ---------------------------------------------------------------------------

FEATURES = [
    "RSI", "MACD_hist", "EMA_20", "ATR", "OBV",
    "news_score", "news_rolling_3d",
    "event_score_max", "is_event",
    "ret_vs_nifty_1d", "ret_vs_nifty_5d",
    "ret_vs_sector_1d", "ret_vs_sector_5d",
    "alpha_strength",
    "vol_spike", "vol_breakout",
    "momentum_diff", "price_pos_20d",
    "PE_Ratio", "ROE",
    # FIX 4: new derived features
    "momentum_strength",
    "volatility_ratio",
]

ALPHA_THRESH = 0.015   # FIX 1: stronger signal filter


def _p(tag, msg):
    print(f"  [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------

def _load_technical():
    if not os.path.exists(TECHNICAL_CSV):
        _p("x", "technical.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(TECHNICAL_CSV)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    df.dropna(subset=["Date", "Close"], inplace=True)
    df.drop_duplicates(subset=["Date", "Stock"], inplace=True)

    before = len(df)
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    _p("OK", f"Technical: {before} → {len(df)} rows ({START_DATE} to {END_DATE})")

    for col in ["Open", "High", "Low", "Close", "Volume", "Return_1d",
                "RSI", "MACD", "MACD_signal", "ATR", "EMA_20", "OBV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_fundamental():
    if not os.path.exists(FUNDAMENTAL_CSV):
        _p("!", "fundamental.csv not found")
        return pd.DataFrame()
    df = pd.read_csv(FUNDAMENTAL_CSV)
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    num_cols = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                "Revenue_Growth", "Profit_Growth"]
    for col in num_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid = df[col].dropna()
            if len(valid) > 10:
                q1, q99 = valid.quantile(0.01), valid.quantile(0.99)
                df[col] = df[col].clip(q1, q99)
    if "Year" not in df.columns:
        df["Year"] = np.nan
    df["Sector"] = df["Stock"].map(SECTOR_MAP).fillna("Unknown")
    return df[["Stock", "Year", "Sector"] + num_cols]


def _load_news():
    """
    Load news.csv and compute per-stock daily average news_score.
    FIX 2: z-score normalize per stock to remove systematic bias.
    """
    if not os.path.exists(NEWS_CSV):
        _p("!", "news.csv not found — news features will be zero")
        return pd.DataFrame()
    df = pd.read_csv(NEWS_CSV)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    df.dropna(subset=["Date", "Stock"], inplace=True)
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]

    if "news_score" not in df.columns:
        if "news_positive" in df.columns and "news_negative" in df.columns:
            df["news_score"] = df["news_positive"] - df["news_negative"]
        else:
            _p("!", "news_score column missing — news features will be zero")
            return pd.DataFrame()

    # Daily average score per (Date, Stock)
    daily = (
        df.groupby(["Date", "Stock"])["news_score"]
          .mean()
          .reset_index()
          .rename(columns={"news_score": "news_score_raw"})
    )

    # FIX 2: per-stock z-score normalization to remove bias
    daily["news_score_daily"] = daily.groupby("Stock")["news_score_raw"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    daily["news_score_daily"] = daily["news_score_daily"].fillna(0.0)
    daily.drop(columns=["news_score_raw"], inplace=True)

    _p("OK", f"News: {len(daily)} (date,stock) pairs | "
              f"normalized score mean={daily['news_score_daily'].mean():.4f} "
              f"std={daily['news_score_daily'].std():.4f}")
    return daily


def _load_events():
    if not os.path.exists(EVENTS_CSV):
        _p("!", "events.csv not found — event features will be zero")
        return pd.DataFrame()
    df = pd.read_csv(EVENTS_CSV)
    df = df.rename(columns={"date": "Date", "symbol": "Stock"})
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    df.dropna(subset=["Date", "Stock"], inplace=True)
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    df.drop_duplicates(subset=["Date", "Stock"], inplace=True)

    keep = ["Date", "Stock", "event_score_max", "is_event", "event_name", "event_category"]
    keep = [c for c in keep if c in df.columns]
    _p("OK", f"Events: {len(df)} rows | event days: {df['is_event'].sum()}")
    return df[keep]


def _load_nifty_features(date_index):
    dates = pd.DatetimeIndex(pd.to_datetime(date_index))
    start = (dates.min() - pd.Timedelta(days=80)).strftime("%Y-%m-%d")
    end   = (dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        import yfinance as yf
        _p("i", "Fetching NIFTY50 (^NSEI) ...")
        nifty = yf.download("^NSEI", start=start, end=end,
                             interval="1d", progress=False, auto_adjust=True)
        if nifty.empty:
            raise ValueError("Empty response")
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.droplevel(1)
        nifty = nifty[["Close"]].copy()
        nifty.index = pd.to_datetime(nifty.index)
        nifty["nifty_ret_1d"]      = nifty["Close"].pct_change(1)
        nifty["nifty_ret_5d"]      = nifty["Close"].pct_change(5)
        delta = nifty["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        nifty["nifty_rsi"]         = 100 - (100 / (1 + gain / (loss + 1e-8)))
        ema20 = nifty["Close"].ewm(span=20, adjust=False).mean()
        nifty["nifty_above_ema20"] = (nifty["Close"] > ema20).astype(int)
        nifty.index = nifty.index.strftime("%Y-%m-%d")
        _p("OK", f"NIFTY features: {len(nifty)} days")
        return nifty[["nifty_ret_1d", "nifty_ret_5d", "nifty_rsi", "nifty_above_ema20"]]
    except Exception as e:
        _p("!", f"NIFTY fetch failed ({e}); zero-filling")
        return pd.DataFrame(0.0, index=date_index,
                            columns=["nifty_ret_1d", "nifty_ret_5d",
                                     "nifty_rsi", "nifty_above_ema20"])


# ---------------------------------------------------------------------------
# PER-STOCK FEATURES
# ---------------------------------------------------------------------------

def _compute_stock_features(df):
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    results = []
    for stock, sdf in df.groupby("Stock", sort=False):
        sdf = sdf.copy()
        c   = sdf["Close"]
        r1  = sdf["Return_1d"] if "Return_1d" in sdf.columns else c.pct_change(1)
        vol = sdf["Volume"]

        sdf["momentum_5d"]   = c.pct_change(5)
        sdf["momentum_10d"]  = c.pct_change(10)
        sdf["momentum_20d"]  = c.pct_change(20)
        sdf["momentum_diff"] = sdf["momentum_5d"] - sdf["momentum_10d"]

        sdf["volatility_10d"] = r1.rolling(10, min_periods=5).std()
        sdf["volatility_20d"] = r1.rolling(20, min_periods=10).std()
        vol20_mean = sdf["volatility_20d"].rolling(20, min_periods=10).mean()
        sdf["vol_breakout"] = sdf["volatility_10d"] / (vol20_mean + 1e-8)

        vol_ma10 = vol.rolling(10, min_periods=5).mean()
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        sdf["vol_spike"] = vol / (vol_ma10 + 1)
        sdf["vol_ratio"] = vol / (vol_ma20 + 1)

        if "RSI" in sdf.columns:
            sdf["rsi_momentum"] = sdf["RSI"] - sdf["RSI"].rolling(5, min_periods=3).mean()
        else:
            sdf["rsi_momentum"] = 0.0

        h20 = c.rolling(20, min_periods=10).max()
        l20 = c.rolling(20, min_periods=10).min()
        sdf["price_pos_20d"] = (c - l20) / (h20 - l20 + 1e-8)

        h252 = c.rolling(252, min_periods=50).max()
        l252 = c.rolling(252, min_periods=50).min()
        sdf["pct_from_52w_high"] = (c - h252) / (h252 + 1e-8)
        sdf["pct_from_52w_low"]  = (c - l252)  / (l252  + 1e-8)

        if all(x in sdf.columns for x in ["High", "Low", "Open"]):
            sdf["close_range_pct"] = (c - sdf["Low"]) / (sdf["High"] - sdf["Low"] + 1e-8)
            sdf["hl_ratio"]        = (sdf["High"] - sdf["Low"]) / (c + 1e-8)
            sdf["gap_pct"]         = (sdf["Open"] - c.shift(1)) / (c.shift(1) + 1e-8)

        if "ATR" in sdf.columns:
            sdf["atr_ratio"] = sdf["ATR"] / (c + 1e-8)

        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        sdf["ema_dist_20"] = (c - ema20) / (ema20 + 1e-8)
        sdf["ema_dist_50"] = (c - ema50) / (ema50 + 1e-8)
        sdf["ema_cross"]   = (ema20 > ema50).astype(int)

        sdf["sharpe_5d"] = (
            r1.rolling(5, min_periods=3).mean() /
            (r1.rolling(5, min_periods=3).std() + 1e-8)
        )

        if "MACD" in sdf.columns and "MACD_signal" in sdf.columns:
            sdf["MACD_hist"] = sdf["MACD"] - sdf["MACD_signal"]

        # FIX 4: New strong derived features
        if "EMA_20" in sdf.columns:
            sdf["momentum_strength"] = c / (sdf["EMA_20"] + 1e-8) - 1
        else:
            sdf["momentum_strength"] = c / (ema20 + 1e-8) - 1

        if "ATR" in sdf.columns:
            sdf["volatility_ratio"] = sdf["ATR"] / (c + 1e-8)
        else:
            sdf["volatility_ratio"] = sdf["volatility_10d"] / (c + 1e-8)

        results.append(sdf)

    out = pd.concat(results, ignore_index=True)
    _p("OK", f"Per-stock features computed: {len(out)} rows")
    return out


def _add_news_features(df, news_daily):
    """Merge news and compute rolling 3-day average per stock."""
    if news_daily.empty:
        df["news_score_daily"] = 0.0
        df["news_rolling_3d"]  = 0.0
        df["news_score"]       = 0.0
        df["has_news"]         = 0
        df["news_spike"]       = 0.0
        return df

    df = df.merge(news_daily, on=["Date", "Stock"], how="left")
    df["news_score_daily"] = df["news_score_daily"].fillna(0.0)
    df["has_news"]         = (df["news_score_daily"] != 0.0).astype(int)

    df = df.sort_values(["Stock", "Date"])
    df["news_rolling_3d"] = (
        df.groupby("Stock")["news_score_daily"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    df["news_spike"] = (
        df.groupby("Stock")["news_score_daily"]
          .transform(lambda x: x - x.rolling(10, min_periods=1).mean())
    )

    # Alias for LSTM and feature list compatibility
    df["news_score"] = df["news_score_daily"]

    non_zero = (df["news_score_daily"] != 0).sum()
    _p("OK", f"News features merged | non-zero rows: {non_zero} "
              f"({non_zero/len(df):.1%} coverage) | "
              f"mean={df['news_score_daily'].mean():.4f}")
    return df


def _add_event_features(df, events_df):
    if events_df.empty:
        df["event_score_max"] = 0.0
        df["is_event"]        = 0
        return df

    merge_cols = ["Date", "Stock", "event_score_max", "is_event"]
    merge_cols = [c for c in merge_cols if c in events_df.columns]
    df = df.merge(events_df[merge_cols], on=["Date", "Stock"], how="left")
    df["event_score_max"] = df["event_score_max"].fillna(0.0)
    df["is_event"]        = df["is_event"].fillna(0).astype(int)

    _p("OK", f"Events merged | event rows: {df['is_event'].sum()}")
    return df


# ---------------------------------------------------------------------------
# SECTOR LEAVE-ONE-OUT FEATURES
# ---------------------------------------------------------------------------

def _add_sector_features(df):
    df = df.copy().sort_values(["Date", "Stock"])

    def _loo(col):
        s   = df.groupby(["Date", "Sector"])[col].transform("sum")
        cnt = df.groupby(["Date", "Sector"])[col].transform("count")
        return (s - df[col]) / (cnt - 1).replace(0, np.nan)

    if "Return_1d" in df.columns:
        df["sector_ret_1d"] = _loo("Return_1d").fillna(0.0)
    else:
        df["sector_ret_1d"] = 0.0

    if "momentum_5d" in df.columns:
        df["sector_ret_5d"] = _loo("momentum_5d").fillna(0.0)
    else:
        df["sector_ret_5d"] = 0.0

    _p("OK", "Sector LOO return features added")
    return df


# ---------------------------------------------------------------------------
# RELATIVE STRENGTH
# ---------------------------------------------------------------------------

def _add_relative_strength(df):
    df = df.copy()
    nifty_1d = df.get("nifty_ret_1d", pd.Series(0.0, index=df.index))
    nifty_5d = df.get("nifty_ret_5d", pd.Series(0.0, index=df.index))
    ret1 = df.get("Return_1d",   pd.Series(0.0, index=df.index))
    mom5 = df.get("momentum_5d", pd.Series(0.0, index=df.index))

    df["ret_vs_nifty_1d"] = ret1 - nifty_1d
    df["ret_vs_nifty_5d"] = mom5 - nifty_5d

    sec1 = df.get("sector_ret_1d", pd.Series(0.0, index=df.index))
    sec5 = df.get("sector_ret_5d", pd.Series(0.0, index=df.index))
    df["ret_vs_sector_1d"] = ret1 - sec1
    df["ret_vs_sector_5d"] = mom5 - sec5

    df["return_vs_sector"] = df["ret_vs_sector_1d"]

    df = df.sort_values(["Stock", "Date"])
    df["alpha_strength"] = (
        df.groupby("Stock")["ret_vs_nifty_1d"]
          .transform(lambda x: x.rolling(5, min_periods=3).mean())
          .fillna(0.0)
    )
    _p("OK", "Relative strength features added")
    return df


# ---------------------------------------------------------------------------
# CROSS-SECTIONAL RANKS
# ---------------------------------------------------------------------------

def _add_cs_ranks(df):
    for col in ["momentum_5d", "vol_spike", "RSI", "ret_vs_nifty_5d"]:
        if col in df.columns:
            df[f"cs_rank_{col}"] = (
                df.groupby("Date")[col]
                  .rank(pct=True, na_option="keep")
                  .fillna(0.5)
            )
    return df


# ---------------------------------------------------------------------------
# ALPHA-BASED LABEL — FIX 1: threshold 0.015
# ---------------------------------------------------------------------------

def _add_alpha_label(df, nifty_fwd_map):
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    df["_fc"] = df.groupby("Stock")["Close"].shift(-5)
    df["stock_ret_5d_fwd"] = (df["_fc"] - df["Close"]) / df["Close"].replace(0, np.nan)
    df.drop(columns=["_fc"], inplace=True)
    df["nifty_ret_5d_fwd"] = df["Date"].map(nifty_fwd_map).fillna(0.0)
    df["alpha_5d"] = df["stock_ret_5d_fwd"] - df["nifty_ret_5d_fwd"]
    df.drop(columns=["stock_ret_5d_fwd", "nifty_ret_5d_fwd"], inplace=True)
    non_null = df["alpha_5d"].notna().sum()
    pos = (df["alpha_5d"] > ALPHA_THRESH).sum()
    neg = (df["alpha_5d"] < -ALPHA_THRESH).sum()
    _p("OK", f"alpha_5d: {non_null} non-null | >{ALPHA_THRESH*100:.1f}%={pos} | <-{ALPHA_THRESH*100:.1f}%={neg}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def merge_all():
    print("\n" + "=" * 62)
    print("  MERGING DATASETS (FIXED v13)")
    print("=" * 62)

    tech = _load_technical()
    if tech.empty:
        print("[ERROR] technical.csv missing or empty")
        return
    _p("OK", f"Technical loaded: {tech.shape}")
    tech["Sector"] = tech["Stock"].map(SECTOR_MAP).fillna("Unknown")

    _p("i", "Computing per-stock rolling features ...")
    tech = _compute_stock_features(tech)

    tech["Year"] = pd.to_datetime(tech["Date"]).dt.year
    fund = _load_fundamental()
    fund_cols = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                 "Revenue_Growth", "Profit_Growth"]
    if not fund.empty:
        merged = tech.merge(fund, on=["Stock", "Year"], how="left", suffixes=("", "_f"))
        merged.drop(columns=[c for c in merged.columns if c.endswith("_f")],
                    inplace=True, errors="ignore")
        merged.sort_values(["Stock", "Date"], inplace=True)
        for col in fund_cols:
            if col in merged.columns:
                merged[col] = (
                    merged.groupby("Stock")[col]
                          .transform(lambda x: x.ffill().bfill())
                )
        _p("OK", f"Fundamentals merged: {merged.shape}")
    else:
        merged = tech.copy()
        for col in fund_cols:
            merged[col] = np.nan
    merged.drop(columns=["Year"], inplace=True, errors="ignore")

    nifty_df = _load_nifty_features(merged["Date"].unique())
    merged = merged.merge(
        nifty_df.reset_index().rename(columns={"index": "Date"}),
        on="Date", how="left"
    )
    for col in ["nifty_ret_1d", "nifty_ret_5d", "nifty_rsi", "nifty_above_ema20"]:
        merged[col] = merged.get(col, pd.Series(0.0, index=merged.index)).fillna(0.0)

    date_nifty = (
        merged[["Date", "nifty_ret_5d"]]
        .drop_duplicates("Date")
        .set_index("Date")["nifty_ret_5d"]
    )
    date_nifty_ts = date_nifty.copy()
    date_nifty_ts.index = pd.to_datetime(date_nifty_ts.index)
    nifty_fwd_series = date_nifty_ts.shift(-5)
    nifty_fwd_map = {k.strftime("%Y-%m-%d"): v for k, v in nifty_fwd_series.items()}

    news_daily = _load_news()
    merged = _add_news_features(merged, news_daily)

    events_df = _load_events()
    merged = _add_event_features(merged, events_df)

    merged = _add_sector_features(merged)
    merged = _add_relative_strength(merged)
    merged = _add_cs_ranks(merged)

    if "PE_Ratio" in merged.columns:
        merged["pe_ratio_rank"] = (
            merged.groupby("Date")["PE_Ratio"]
                  .rank(pct=True, na_option="keep")
                  .fillna(0.5)
        )
    else:
        merged["pe_ratio_rank"] = 0.5

    merged = _add_alpha_label(merged, nifty_fwd_map)

    before = len(merged)
    merged.dropna(subset=["Close"], inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    if "atr_ratio" in merged.columns:
        thr = merged["atr_ratio"].quantile(0.20)
        merged = merged[merged["atr_ratio"] >= thr]
        _p("OK", f"Low-vol filter: {before} → {len(merged)} rows")

    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    const_cols = [c for c in num_cols if merged[c].nunique() <= 1]
    if const_cols:
        _p("!", f"Dropping constant columns: {const_cols}")
        merged.drop(columns=const_cols, inplace=True)

    merged.sort_values(["Stock", "Date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)

    print(f"\n[OK] Saved: {MERGED_CSV}")
    print(f"Shape: {merged.shape} | Columns: {len(merged.columns)}")
    print(f"Date range: {merged['Date'].min()} → {merged['Date'].max()}")
    pre2020 = (merged["Date"] < "2020-01-01").sum()
    print(f"Pre-2020 rows: {pre2020}  (should be 0)")

    alpha_valid = merged["alpha_5d"].notna().sum()
    pos  = (merged["alpha_5d"] > ALPHA_THRESH).sum()
    neg  = (merged["alpha_5d"] < -ALPHA_THRESH).sum()
    print(f"\n  Alpha label (threshold={ALPHA_THRESH}) → total={alpha_valid}  UP={pos}  DOWN={neg}")
    if (pos + neg) > 0:
        print(f"  Balance: UP%={pos/(pos+neg)*100:.1f}%")

    # Feature presence check
    print("\n  FEATURES list presence check:")
    for f in FEATURES:
        status = "✓" if f in merged.columns else "✗ MISSING"
        print(f"    {f:30s} {status}")

    # Step 8 — M&M validation
    mm = merged[merged["Stock"] == "M&M"]
    print(f"\n  === M&M Validation ===")
    print(f"    Rows:                  {len(mm)}")
    if "news_score" in mm.columns:
        nz = (mm["news_score"] != 0).sum()
        mn = mm["news_score"].mean()
        print(f"    news_score != 0:       {nz}  (mean={mn:.4f})")
    else:
        print(f"    news_score:            MISSING")
    if "is_event" in mm.columns:
        print(f"    event rows:            {mm['is_event'].sum()}")
    else:
        print(f"    is_event:              MISSING")
    if "ROE" in mm.columns:
        print(f"    ROE non-null:          {mm['ROE'].notna().sum()}")
    if "PE_Ratio" in mm.columns:
        print(f"    PE_Ratio non-null:     {mm['PE_Ratio'].notna().sum()}")
    if "momentum_strength" in mm.columns:
        print(f"    momentum_strength OK:  ✓")
    if "volatility_ratio" in mm.columns:
        print(f"    volatility_ratio OK:   ✓")


if __name__ == "__main__":
    merge_all()