# =============================================================================
# features/merge_features.py  (REDESIGNED v10 — Alpha-Based Pipeline)
#
# KEY CHANGES vs v8/v9:
#
#   1. [TARGET] Alpha-based label replaces raw direction:
#      alpha_5d = stock_ret_5d - nifty_ret_5d  (forward-looking)
#      target = 1 if alpha_5d > 0, else 0
#      Only rows where abs(alpha_5d) > 1% are kept (noise filter).
#      This removes market-beta noise and predicts genuine outperformance.
#
#   2. [FEATURES] Strong stock-specific features:
#      - Relative strength vs NIFTY: ret_vs_nifty_1d, ret_vs_nifty_5d
#      - Relative strength vs sector: ret_vs_sector_1d, ret_vs_sector_5d
#      - Alpha stability: alpha_strength (rolling mean of ret_vs_nifty_1d)
#      - Volume spike: vol_spike = Volume / rolling_mean(Volume, 10)
#      - Vol breakout: current_vol / mean_vol_20d
#      - Momentum: momentum_5d, momentum_10d, momentum_diff
#      - Price position in 20-day high-low range
#
#   3. [REMOVED] Weak/noisy features:
#      - event_* columns (sparse, near-zero signal)
#      - news_* columns (inconsistent coverage <15%)
#      - sector_encoded (redundant with sector return features)
#      - Raw price lags and redundant indicators
#      Feature count: ~30 clean, meaningful features.
#
#   4. [CLEANING]
#      - Removes low-volatility rows (bottom 20% by ATR/Close)
#      - Drops NaN / infinite values in core columns
#
#   5. [LEAKAGE-FREE]
#      - All rolling windows are per-stock, past-only
#      - alpha_5d uses future data and is stored as a label-only column
#      - It is stripped from features in train.py after label extraction
# =============================================================================

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, MERGED_CSV,
    SECTOR_MAP,
)


def _p(tag, msg):
    print(f"  [{tag}] {msg}")


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
    for col in ["Open", "High", "Low", "Close", "Volume", "Return_1d",
                "RSI", "MACD", "MACD_signal", "ATR", "EMA_20"]:
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


def _load_nifty_features(date_index):
    """Fetch NIFTY50 features; falls back to zeros gracefully."""
    dates = pd.DatetimeIndex(pd.to_datetime(date_index))
    start = dates.min() - pd.Timedelta(days=80)
    end   = dates.max() + pd.Timedelta(days=5)
    try:
        import yfinance as yf
        _p("i", "Fetching NIFTY50 (^NSEI) ...")
        nifty = yf.download("^NSEI", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             interval="1d", progress=False, auto_adjust=True)
        if nifty.empty:
            raise ValueError("Empty response")
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.droplevel(1)
        nifty = nifty[["Close"]].copy()
        nifty.index = pd.to_datetime(nifty.index)
        nifty["nifty_ret_1d"] = nifty["Close"].pct_change(1)
        nifty["nifty_ret_5d"] = nifty["Close"].pct_change(5)
        delta = nifty["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        nifty["nifty_rsi"] = 100 - (100 / (1 + gain / (loss + 1e-8)))
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
        sdf  = sdf.copy()
        c    = sdf["Close"]
        r1   = sdf["Return_1d"] if "Return_1d" in sdf.columns else c.pct_change(1)
        vol  = sdf["Volume"]

        # Momentum
        sdf["momentum_5d"]   = c.pct_change(5)
        sdf["momentum_10d"]  = c.pct_change(10)
        sdf["momentum_20d"]  = c.pct_change(20)
        sdf["momentum_diff"] = sdf["momentum_5d"] - sdf["momentum_10d"]

        # Volatility
        sdf["volatility_10d"] = r1.rolling(10, min_periods=5).std()
        sdf["volatility_20d"] = r1.rolling(20, min_periods=10).std()
        vol20_mean = sdf["volatility_20d"].rolling(20, min_periods=10).mean()
        sdf["vol_breakout"] = sdf["volatility_10d"] / (vol20_mean + 1e-8)

        # Volume spike
        vol_ma10 = vol.rolling(10, min_periods=5).mean()
        vol_ma20 = vol.rolling(20, min_periods=10).mean()
        sdf["vol_spike"] = vol / (vol_ma10 + 1)
        sdf["vol_ratio"] = vol / (vol_ma20 + 1)

        # RSI momentum
        if "RSI" in sdf.columns:
            sdf["rsi_momentum"] = sdf["RSI"] - sdf["RSI"].rolling(5, min_periods=3).mean()
        else:
            sdf["rsi_momentum"] = 0.0

        # Price position in 20-day range
        h20 = c.rolling(20, min_periods=10).max()
        l20 = c.rolling(20, min_periods=10).min()
        sdf["price_pos_20d"] = (c - l20) / (h20 - l20 + 1e-8)

        # 52-week position
        h252 = c.rolling(252, min_periods=50).max()
        l252 = c.rolling(252, min_periods=50).min()
        sdf["pct_from_52w_high"] = (c - h252) / (h252 + 1e-8)
        sdf["pct_from_52w_low"]  = (c - l252)  / (l252  + 1e-8)

        # Intraday structure
        if all(x in sdf.columns for x in ["High", "Low", "Open"]):
            sdf["close_range_pct"] = (c - sdf["Low"]) / (sdf["High"] - sdf["Low"] + 1e-8)
            sdf["hl_ratio"]        = (sdf["High"] - sdf["Low"]) / (c + 1e-8)
            sdf["gap_pct"]         = (sdf["Open"] - c.shift(1)) / (c.shift(1) + 1e-8)

        # ATR ratio
        if "ATR" in sdf.columns:
            sdf["atr_ratio"] = sdf["ATR"] / (c + 1e-8)

        # EMA trend
        ema20 = c.ewm(span=20, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()
        sdf["ema_dist_20"] = (c - ema20) / (ema20 + 1e-8)
        sdf["ema_dist_50"] = (c - ema50) / (ema50 + 1e-8)
        sdf["ema_cross"]   = (ema20 > ema50).astype(int)

        # Sharpe-5d
        sdf["sharpe_5d"] = (
            r1.rolling(5, min_periods=3).mean() /
            (r1.rolling(5, min_periods=3).std() + 1e-8)
        )

        results.append(sdf)

    out = pd.concat(results, ignore_index=True)
    _p("OK", f"Per-stock features computed: {len(out)} rows")
    return out


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
# RELATIVE STRENGTH (stock vs NIFTY and vs sector)
# ---------------------------------------------------------------------------

def _add_relative_strength(df):
    df = df.copy()

    # vs NIFTY
    nifty_1d = df.get("nifty_ret_1d", pd.Series(0.0, index=df.index))
    nifty_5d = df.get("nifty_ret_5d", pd.Series(0.0, index=df.index))
    ret1  = df.get("Return_1d",   pd.Series(0.0, index=df.index))
    mom5  = df.get("momentum_5d", pd.Series(0.0, index=df.index))

    df["ret_vs_nifty_1d"] = ret1  - nifty_1d
    df["ret_vs_nifty_5d"] = mom5  - nifty_5d

    # vs sector
    sec1 = df.get("sector_ret_1d", pd.Series(0.0, index=df.index))
    sec5 = df.get("sector_ret_5d", pd.Series(0.0, index=df.index))
    df["ret_vs_sector_1d"] = ret1 - sec1
    df["ret_vs_sector_5d"] = mom5 - sec5

    # Alpha stability (rolling mean of daily alpha vs NIFTY)
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
# ALPHA-BASED LABEL (forward-looking — NOT a feature)
# ---------------------------------------------------------------------------

def _add_alpha_label(df, nifty_fwd_map):
    """
    alpha_5d = forward_5d_stock_return - forward_5d_nifty_return.
    nifty_fwd_map: {date_str: nifty_5d_forward_return}
    alpha_5d is stored in the CSV for train.py to use as a label source.
    """
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # Forward 5-day stock return
    df["_fc"] = df.groupby("Stock")["Close"].shift(-5)
    df["stock_ret_5d_fwd"] = (df["_fc"] - df["Close"]) / df["Close"].replace(0, np.nan)
    df.drop(columns=["_fc"], inplace=True)

    # Map forward NIFTY return to each row date
    df["nifty_ret_5d_fwd"] = df["Date"].map(nifty_fwd_map).fillna(0.0)

    df["alpha_5d"] = df["stock_ret_5d_fwd"] - df["nifty_ret_5d_fwd"]
    df.drop(columns=["stock_ret_5d_fwd", "nifty_ret_5d_fwd"], inplace=True)

    non_null = df["alpha_5d"].notna().sum()
    pos = (df["alpha_5d"] > 0.01).sum()
    neg = (df["alpha_5d"] < -0.01).sum()
    _p("OK", f"alpha_5d: {non_null} non-null | >+1%={pos} | <-1%={neg}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def merge_all():
    print("\n" + "=" * 62)
    print("  MERGING DATASETS (REDESIGNED v10 — Alpha-Based Pipeline)")
    print("=" * 62)

    # 1. Technical
    tech = _load_technical()
    if tech.empty:
        print("[ERROR] technical.csv missing or empty")
        return
    _p("OK", f"Technical loaded: {tech.shape}")
    tech["Sector"] = tech["Stock"].map(SECTOR_MAP).fillna("Unknown")

    # 2. Per-stock features
    _p("i", "Computing per-stock rolling features ...")
    tech = _compute_stock_features(tech)

    # 3. Merge fundamentals
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

    # 4. NIFTY features
    nifty_df = _load_nifty_features(merged["Date"].unique())
    merged = merged.merge(
        nifty_df.reset_index().rename(columns={"index": "Date"}),
        on="Date", how="left"
    )
    for col in ["nifty_ret_1d", "nifty_ret_5d", "nifty_rsi", "nifty_above_ema20"]:
        merged[col] = merged.get(col, pd.Series(0.0, index=merged.index)).fillna(0.0)

    # Build forward NIFTY map for alpha label
    date_nifty = (
        merged[["Date", "nifty_ret_5d"]]
        .drop_duplicates("Date")
        .set_index("Date")["nifty_ret_5d"]
    )
    dates_sorted = pd.to_datetime(date_nifty.index).sort_values()
    date_nifty_ts = date_nifty.copy()
    date_nifty_ts.index = pd.to_datetime(date_nifty_ts.index)
    nifty_fwd_series = date_nifty_ts.shift(-5)   # 5 trading days forward
    nifty_fwd_map = {k.strftime("%Y-%m-%d"): v
                     for k, v in nifty_fwd_series.items()}

    # 5. Sector LOO features
    merged = _add_sector_features(merged)

    # 6. Relative strength
    merged = _add_relative_strength(merged)

    # 7. Cross-sectional ranks
    merged = _add_cs_ranks(merged)

    # 8. PE rank
    if "PE_Ratio" in merged.columns:
        merged["pe_ratio_rank"] = (
            merged.groupby("Date")["PE_Ratio"]
                  .rank(pct=True, na_option="keep")
                  .fillna(0.5)
        )
    else:
        merged["pe_ratio_rank"] = 0.5

    # 9. Alpha label (forward-looking)
    merged = _add_alpha_label(merged, nifty_fwd_map)

    # 10. Data cleaning
    before = len(merged)
    merged.dropna(subset=["Close"], inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove low-volatility rows (bottom 20% by ATR ratio)
    if "atr_ratio" in merged.columns:
        thr = merged["atr_ratio"].quantile(0.20)
        merged = merged[merged["atr_ratio"] >= thr]
        _p("OK", f"Low-vol filter: {before} → {len(merged)} rows removed "
                  f"{before - len(merged)}")

    # Fill NaN with medians
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    medians  = merged[num_cols].median()
    merged[num_cols] = merged[num_cols].fillna(medians)

    # Drop constant columns
    const_cols = [c for c in num_cols if merged[c].nunique() <= 1]
    if const_cols:
        _p("!", f"Dropping constant columns: {const_cols}")
        merged.drop(columns=const_cols, inplace=True)

    # 11. Save
    merged.sort_values(["Stock", "Date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    merged.to_csv(MERGED_CSV, index=False)

    print(f"\n[OK] Saved: {MERGED_CSV}")
    print(f"Shape: {merged.shape} | Columns: {len(merged.columns)}")

    alpha_valid = merged["alpha_5d"].notna().sum()
    pos  = (merged["alpha_5d"] > 0.01).sum()
    neg  = (merged["alpha_5d"] < -0.01).sum()
    print(f"\n  Alpha label  →  total={alpha_valid}  UP(>+1%)={pos}  DOWN(<-1%)={neg}")
    print(f"  Balance: UP%={pos/(pos+neg)*100:.1f}%" if (pos+neg) > 0 else "")

    print("\n  Feature check:")
    key_feats = ["ret_vs_nifty_1d", "ret_vs_nifty_5d", "ret_vs_sector_1d",
                 "ret_vs_sector_5d", "alpha_strength", "vol_spike",
                 "vol_breakout", "momentum_diff", "price_pos_20d",
                 "atr_ratio", "alpha_5d"]
    for f in key_feats:
        status = "✓" if f in merged.columns else "✗ MISSING"
        print(f"    {f:30s} {status}")
    print("\n  Sector distribution:")
    print(merged["Sector"].value_counts().to_string())


if __name__ == "__main__":
    merge_all()
