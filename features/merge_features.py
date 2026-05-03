"""
features/merge_features.py  — BUG FIXES: 1, 3, 4, 7

BUG-1 (CRITICAL): news_score was always 0.0 (FinBERT never run on news.csv).
  FIX: Read finbert_scores.csv, compute news_score = finbert_pos - finbert_neg.

BUG-3: Revenue_Growth / Profit_Growth NaN filled as 0 (69.8% rows wrong).
  FIX: Sector-year median imputation + has_fundamental_data binary flag.

BUG-4: MACD_signal had 320 partial-NaN rows (partial-NaN pass-through).
  FIX: Recompute MACD per stock whenever any NaN exists in that stock.

FIX-7: Add ret_lag_1d, ret_lag_3d, ret_lag_5d (shift>=1, strictly no leakage).
"""

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

# BUG-1 FIX: path to real FinBERT scores
FINBERT_SCORES_PATH = os.path.join(os.path.dirname(NEWS_CSV), "finbert_scores.csv")

TICKER_ALIAS = {
    "M&M": "M&M", "M_M": "M&M", "M-M": "M&M", "MM": "M&M",
    "M&M.NS": "M&M", "M_M.NS": "M&M",
    "BAJAJ-AUTO": "BAJAJ-AUTO", "BAJAJ_AUTO": "BAJAJ-AUTO",
    "BAJAJAUTO": "BAJAJ-AUTO", "BAJAJ AUTO": "BAJAJ-AUTO",
    "BAJAJ-AUTO.NS": "BAJAJ-AUTO",
    "HDFC BANK": "HDFCBANK", "HDFC-BANK": "HDFCBANK",
}

UP_THRESH   =  0.015
DOWN_THRESH = -0.015


def normalise_ticker(series):
    s = series.astype(str).str.replace(r"\.(NS|BO)$", "", regex=True).str.strip()
    return s.map(lambda t: TICKER_ALIAS.get(t, t))


def _remove_constant_cols(df, feature_cols):
    drop = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if drop:
        log.warning(f"Dropping constant cols: {drop}")
    return [c for c in feature_cols if c not in drop]


_NIFTY_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "merged", "_nifty_cache.csv",
)


def _load_nifty():
    raw = pd.DataFrame()
    try:
        log.info("Downloading ^NSEI from yfinance ...")
        raw = yf.download("^NSEI", start=DATE_START, end=DATE_END,
                          auto_adjust=True, progress=False)
    except Exception as e:
        log.warning(f"yfinance fetch failed: {e}")

    if raw is None or raw.empty:
        if os.path.exists(_NIFTY_CACHE):
            log.warning(f"Falling back to cached nifty file: {_NIFTY_CACHE}")
            return pd.read_csv(_NIFTY_CACHE, parse_dates=["Date"])
        raise RuntimeError("yfinance returned empty data for ^NSEI and no cache available.")

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
    log.info(f"Nifty: {nifty.shape}")
    try:
        os.makedirs(os.path.dirname(_NIFTY_CACHE), exist_ok=True)
        nifty.to_csv(_NIFTY_CACHE, index=False)
    except Exception:
        pass
    return nifty


# ── BUG-1 FIX: load real FinBERT scores ──────────────────────────────────────
def _load_finbert_scores():
    if not os.path.exists(FINBERT_SCORES_PATH):
        log.warning(
            f"finbert_scores.csv NOT found at {FINBERT_SCORES_PATH}. "
            "Run models/finbert/infer_news.py first. "
            "Falling back to news_score=0.0 for all rows."
        )
        return None
    fb = pd.read_csv(FINBERT_SCORES_PATH, parse_dates=["Date"])
    fb["Stock"] = normalise_ticker(fb["Stock"])
    fb = (
        fb.groupby(["Date", "Stock"], as_index=False)
          [["finbert_pos", "finbert_neg", "finbert_neu"]]
          .mean()
    )
    fb["news_score_fb"] = fb["finbert_pos"] - fb["finbert_neg"]
    log.info(
        f"BUG-1 FIX: FinBERT scores loaded — {len(fb):,} rows | "
        f"{fb['Stock'].nunique()} tickers | "
        f"mean_news_score={fb['news_score_fb'].mean():.4f}"
    )
    return fb


# ─────────────────────────────────────────────────────────────────────────────
def merge_all():
    print("\n" + "=" * 60)
    print("  MERGE FEATURES  (BUG-1, BUG-3, BUG-4, FIX-7 applied)")
    print("=" * 60)

    if not os.path.exists(TECHNICAL_CSV):
        raise FileNotFoundError(
            f"Missing {TECHNICAL_CSV}. Run data_collection/build_technical.py first.")

    # ── 1. Technical ─────────────────────────────────────────────────────────
    tech = pd.read_csv(TECHNICAL_CSV)
    tech["Date"]  = pd.to_datetime(tech["Date"])
    tech["Stock"] = normalise_ticker(tech["Stock"])
    tech = tech[(tech["Date"] >= DATE_START) & (tech["Date"] <= DATE_END)]
    tech = tech.sort_values(["Stock", "Date"])
    tech = tech.drop_duplicates(subset=["Stock", "Date"], keep="last")
    log.info(f"Technical loaded: {tech.shape}")

    # ── BUG-4 FIX: recompute MACD_signal per stock when any NaN present ───────
    # Old code: only recomputed if entire column was all-NaN — partial NaN
    # rows (first ~26 rows per stock before EMA-26 warms up) passed through
    # and were later zero-filled, corrupting the signal.
    if "MACD_signal" in tech.columns:
        stocks_with_nan = tech[tech["MACD_signal"].isna()]["Stock"].unique()
    else:
        stocks_with_nan = tech["Stock"].unique()

    if len(stocks_with_nan) > 0:
        log.info(f"BUG-4 FIX: Recomputing MACD for {len(stocks_with_nan)} stocks ...")
        fixed, ok = [], []
        for stk, sub in tech.groupby("Stock"):
            if stk in stocks_with_nan:
                sub = sub.copy().sort_values("Date")
                ema12 = sub["Close"].ewm(span=12, adjust=False).mean()
                ema26 = sub["Close"].ewm(span=26, adjust=False).mean()
                macd  = ema12 - ema26
                sub["MACD_signal"] = macd.ewm(span=9, adjust=False).mean()
                sub["MACD_hist"]   = macd - sub["MACD_signal"]
                fixed.append(sub)
            else:
                ok.append(sub)
        tech = pd.concat(fixed + ok).sort_values(["Stock","Date"]).reset_index(drop=True)
        log.info(f"  Remaining NaN in MACD_signal: {tech['MACD_signal'].isna().sum()} "
                 f"(first warm-up rows — acceptable)")

    g = tech.groupby("Stock", group_keys=False)

    # ── Momentum features ─────────────────────────────────────────────────────
    tech["momentum_5d"]       = g["Close"].transform(lambda x: x.pct_change(5))
    tech["momentum_10d"]      = g["Close"].transform(lambda x: x.pct_change(10))
    tech["momentum_diff"]     = tech["momentum_5d"] - tech["momentum_10d"]
    tech["momentum_strength"] = (
        tech.groupby("Stock")["momentum_5d"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # ── FIX-7: Lag features (strictly no look-ahead, shift >= 1) ─────────────
    tech["ret_lag_1d"] = g["Return_1d"].transform(lambda x: x.shift(1))
    tech["ret_lag_3d"] = g["Return_1d"].transform(lambda x: x.shift(3))
    tech["ret_lag_5d"] = g["Return_1d"].transform(lambda x: x.shift(5))
    log.info("FIX-7: ret_lag_1d / ret_lag_3d / ret_lag_5d added")

    # ── Volatility ────────────────────────────────────────────────────────────
    tech["volatility_ratio"] = tech["ATR"] / (tech["Close"].abs() + 1e-8)
    tech["_daily_ret"] = g["Close"].transform(lambda x: x.pct_change(1))
    tech["hist_vol_20d"] = (
        tech.groupby("Stock")["_daily_ret"]
            .transform(lambda x: x.rolling(20, min_periods=5).std())
    )
    tech.drop(columns=["_daily_ret"], inplace=True)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = g["Close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    bb_std = g["Close"].transform(lambda x: x.rolling(20, min_periods=5).std())
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    tech["bb_width"] = (bb_upper - bb_lower) / (bb_mid.abs() + 1e-8)
    tech["bb_pct"]   = (tech["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    tech["atr_norm_range"] = (tech["High"] - tech["Low"]) / (tech["ATR"] + 1e-8)

    # ── Volume features ───────────────────────────────────────────────────────
    vol_ma20    = g["Volume"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    close_max20 = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).max())
    close_ma20  = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    tech["vol_spike"]     = (tech["Volume"] > vol_ma20 * 1.5).astype(int)
    tech["vol_breakout"]  = (tech["Close"]  > close_max20.shift(1)).astype(int)
    tech["price_pos_20d"] = tech["Close"] / (close_ma20 + 1e-8)

    # ── MACD / EMA fallback ───────────────────────────────────────────────────
    ema12 = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd  = ema12 - ema26
    if "MACD_signal" not in tech.columns or tech["MACD_signal"].isna().all():
        tech["MACD_signal"] = macd.groupby(tech["Stock"]).transform(
            lambda x: x.ewm(span=9, adjust=False).mean())
    if "MACD_hist" not in tech.columns or tech["MACD_hist"].isna().all():
        tech["MACD_hist"] = macd - tech["MACD_signal"]
    if "EMA_20" not in tech.columns or tech["EMA_20"].isna().all():
        tech["EMA_20"] = g["Close"].transform(
            lambda x: x.ewm(span=20, adjust=False).mean())
    log.info("Technical-derived features computed")

    # ── 2. Nifty index features ───────────────────────────────────────────────
    nifty = _load_nifty()
    tech = tech.merge(nifty[["Date", "nifty_ret_1d", "nifty_ret_5d"]],
                      on="Date", how="left")
    tech["ret_vs_nifty_1d"] = tech["Return_1d"]   - tech["nifty_ret_1d"]
    tech["ret_vs_nifty_5d"] = tech["momentum_5d"] - tech["nifty_ret_5d"]
    log.info("Nifty market features merged")

    # ── 3. Sector features ────────────────────────────────────────────────────
    tech["Sector"]         = tech["Stock"].map(SECTOR_MAP).fillna("Unknown")
    tech["sector_encoded"] = tech["Sector"].map(SECTOR_TO_CODE).fillna(-1).astype(int)
    sector_daily = (
        tech.groupby(["Date", "Sector"])["Return_1d"]
            .mean().rename("sector_ret_1d_raw").reset_index()
    )
    sector_daily = sector_daily.sort_values(["Sector", "Date"])
    sector_daily["sector_ret_1d"] = (
        sector_daily.groupby("Sector")["sector_ret_1d_raw"]
                    .transform(lambda x: x.shift(1))
    )
    sector_daily["sector_ret_5d"] = (
        sector_daily.groupby("Sector")["sector_ret_1d_raw"]
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    sector_daily.drop(columns=["sector_ret_1d_raw"], inplace=True)
    tech = tech.merge(sector_daily, on=["Date", "Sector"], how="left")
    tech["sector_ret_1d"]    = tech["sector_ret_1d"].fillna(0.0)
    tech["sector_ret_5d"]    = tech["sector_ret_5d"].fillna(0.0)
    tech["return_vs_sector"] = tech["Return_1d"].fillna(0.0) - tech["sector_ret_1d"]
    tech["sector_rel_momentum"] = tech["momentum_10d"] - tech["sector_ret_5d"]
    log.info("Sector features merged")

    # ── 4. Fundamentals (BUG-3 FIX) ──────────────────────────────────────────
    if os.path.exists(FUNDAMENTAL_CSV):
        fund = pd.read_csv(FUNDAMENTAL_CSV)
        fund["Stock"] = normalise_ticker(fund["Stock"])
        fund["Date"]  = pd.to_datetime(fund["Year"].astype(int).astype(str) + "-12-31")
        fund = fund.sort_values(["Stock", "Date"]).drop_duplicates(
            subset=["Stock", "Date"], keep="last")

        df = pd.merge_asof(
            tech.sort_values("Date"),
            fund[["Stock", "Date", "PE_Ratio", "ROE",
                  "Revenue_Growth", "Profit_Growth"]].sort_values("Date"),
            on="Date", by="Stock", direction="backward",
        )

        # BUG-3 FIX: flag rows with real fundamental data before any imputation
        df["has_fundamental_data"] = (
            df["PE_Ratio"].notna() & df["ROE"].notna()
        ).astype(int)

        # BUG-3 FIX: sector-year median imputation (NOT global zero fill)
        df["_year"] = df["Date"].dt.year
        fund_cols = ["PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]
        for col in fund_cols:
            nan_mask = df[col].isna()
            if nan_mask.any():
                # Pass 1: sector + year median from real rows
                sec_yr_med = (
                    df[~nan_mask]
                    .groupby(["Sector", "_year"])[col]
                    .median()
                    .rename("_imp")
                    .reset_index()
                )
                df = df.merge(sec_yr_med, on=["Sector", "_year"], how="left")
                still = df[col].isna()
                df.loc[still & df["_imp"].notna(), col] = df.loc[
                    still & df["_imp"].notna(), "_imp"
                ]
                df.drop(columns=["_imp"], inplace=True)

                # Pass 2: global year median for any remaining NaN
                still2 = df[col].isna()
                if still2.any():
                    yr_med = df[~df[col].isna()].groupby("_year")[col].median()
                    df.loc[still2, col] = df.loc[still2, "_year"].map(yr_med)

                # Pass 3: absolute last-resort zero fill
                df[col] = df[col].fillna(0.0)

        df.drop(columns=["_year"], inplace=True)
        rev_zero_pct = (df["Revenue_Growth"] == 0.0).mean() * 100
        log.info(
            f"BUG-3 FIX: Fundamentals imputed. "
            f"Revenue_Growth=0 rows: {rev_zero_pct:.1f}% (was 69.8%). "
            f"has_fundamental_data=1: {int(df['has_fundamental_data'].sum())} rows."
        )
    else:
        df = tech.copy()
        for c in ["PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]:
            df[c] = np.nan
        df["has_fundamental_data"] = 0
        log.warning("Fundamental CSV missing — fundamentals set to NaN")

    # ── 5. News — BUG-1 FIX: use finbert_scores.csv, not news.csv ─────────────
    fb_scores = _load_finbert_scores()

    if fb_scores is not None:
        df = df.merge(
            fb_scores[["Date", "Stock", "news_score_fb",
                        "finbert_pos", "finbert_neg", "finbert_neu"]],
            on=["Date", "Stock"], how="left",
        )
        df["news_score"] = df["news_score_fb"].fillna(0.0)
        df.drop(columns=["news_score_fb"], inplace=True)
        covered = df["finbert_pos"].notna().sum()
        log.info(
            f"BUG-1 FIX: Real FinBERT news_score merged. "
            f"Coverage: {covered}/{len(df)} ({covered/len(df)*100:.1f}%). "
            f"Mean news_score={df['news_score'].mean():.4f}"
        )
    else:
        # Fallback: check if news.csv has real variation (not all 0.333 / 0.0)
        if os.path.exists(NEWS_CSV):
            news = pd.read_csv(NEWS_CSV)
            news["Date"]  = pd.to_datetime(news["Date"])
            news["Stock"] = normalise_ticker(news["Stock"])
            if "news_score" in news.columns:
                n_unique = news["news_score"].dropna().nunique()
                if n_unique <= 2:
                    log.warning(
                        f"BUG-1 STILL PRESENT: news.csv[news_score] has only "
                        f"{n_unique} unique values — FinBERT was never run. "
                        "news_score=0.0. Run models/finbert/infer_news.py to fix."
                    )
                    df["news_score"] = 0.0
                else:
                    news = news[["Date","Stock","news_score"]]
                    news = news.groupby(["Date","Stock"],
                                        as_index=False)["news_score"].mean()
                    df = df.merge(news, on=["Date","Stock"], how="left")
                    df["news_score"] = df["news_score"].fillna(0.0)
            else:
                df["news_score"] = 0.0
        else:
            df["news_score"] = 0.0
        for col in ["finbert_pos", "finbert_neg", "finbert_neu"]:
            df[col] = 1.0 / 3.0

    df["has_news"] = (df["news_score"] != 0.0).astype(int)
    df = df.sort_values(["Stock", "Date"])
    df["news_rolling_3d"] = (
        df.groupby("Stock")["news_score"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["news_decay"] = (
        df.groupby("Stock")["news_score"]
          .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    log.info(f"News features | has_news rows={int(df['has_news'].sum())}")

    # ── 6. Events ─────────────────────────────────────────────────────────────
    if os.path.exists(EVENTS_CSV):
        events = pd.read_csv(EVENTS_CSV)
        events = events.rename(columns={"date": "Date", "symbol": "Stock"})
        events["Date"]  = pd.to_datetime(events["Date"])
        events["Stock"] = normalise_ticker(events["Stock"])
        events_agg = (
            events.groupby(["Date", "Stock"], as_index=False)
                  .agg(event_score_max=("event_score_max","max"),
                       is_event=("is_event","max"))
        )
        df = df.merge(events_agg, on=["Date","Stock"], how="left")
    else:
        df["event_score_max"] = 0.0
        df["is_event"]        = 0
        log.warning("Events CSV missing — event columns set to 0")

    df["is_event"]        = df["is_event"].fillna(0).astype(int)
    df["event_score_max"] = df["event_score_max"].fillna(0.0)
    df["event_strength"]  = np.log1p(df["event_score_max"].clip(lower=0))
    df["event_impact_decay"] = (
        df.groupby("Stock")["event_strength"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    log.info(f"Events merged | event_rows={int(df['is_event'].sum())}")

    df["alpha_strength"] = (
        df.groupby("Stock")["Close"]
          .transform(lambda x: x.pct_change(5).abs())
    )

    # ── 7. Label generation ───────────────────────────────────────────────────
    log.info(f"LABEL_HORIZON={LABEL_HORIZON} | UP>{UP_THRESH*100:.1f}% DOWN<{DOWN_THRESH*100:.1f}%")
    df["_fwd_close"] = df.groupby("Stock")["Close"].shift(-LABEL_HORIZON)
    df["_fwd_ret"]   = (df["_fwd_close"] - df["Close"]) / df["Close"].replace(0, np.nan)
    df["label"] = np.where(
        df["_fwd_ret"] > UP_THRESH, 1,
        np.where(df["_fwd_ret"] < DOWN_THRESH, -1, 0)
    )
    df.loc[df["_fwd_ret"].isna(), "label"] = np.nan
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    valid = len(df)
    up   = int((df["label"] ==  1).sum())
    flat = int((df["label"] ==  0).sum())
    down = int((df["label"] == -1).sum())
    log.info(f"Labels: total={valid} UP={up}({up/valid*100:.1f}%) "
             f"FLAT={flat}({flat/valid*100:.1f}%) DOWN={down}({down/valid*100:.1f}%)")

    # ── 8. Clean up leakage columns ───────────────────────────────────────────
    drop_cols = ["_fwd_close", "_fwd_ret", "Direction", "Year", "Sector"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # BUG-3 FIX: do NOT zero-fill fundamental columns — already imputed above.
    # Zero-fill only non-fundamental numeric columns (technical, event, etc.)
    fund_cols_set = {"PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"}
    safe_fill = [c for c in num_cols if c not in fund_cols_set]
    df[safe_fill] = df[safe_fill].fillna(0)
    df[[c for c in fund_cols_set if c in df.columns]] = (
        df[[c for c in fund_cols_set if c in df.columns]].fillna(0)
    )
    df[str_cols] = df[str_cols].fillna("")

    # ── 9. Feature validation ─────────────────────────────────────────────────
    meta_cols  = ["Date", "Stock", "label"]
    raw_price  = ["Open", "High", "Low", "Close", "Volume"]
    feat_cands = [c for c in df.columns if c not in meta_cols + raw_price]
    feat_cands = _remove_constant_cols(df, feat_cands)
    log.info(f"Final feature count: {len(feat_cands)}")

    mandatory = [
        "RSI", "MACD_hist", "MACD_signal", "EMA_20", "ATR", "OBV",
        "momentum_5d", "momentum_10d", "momentum_diff", "momentum_strength",
        "ret_lag_1d", "ret_lag_3d", "ret_lag_5d",          # FIX-7
        "volatility_ratio", "hist_vol_20d",
        "bb_width", "bb_pct", "atr_norm_range",
        "vol_spike", "vol_breakout", "price_pos_20d",
        "news_score", "news_rolling_3d", "news_decay", "has_news",  # BUG-1 real scores
        "event_strength", "event_impact_decay", "event_score_max", "is_event",
        "PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth",
        "has_fundamental_data",                             # BUG-3 flag
        "alpha_strength",
        "nifty_ret_1d", "nifty_ret_5d",
        "ret_vs_nifty_1d", "ret_vs_nifty_5d",
        "sector_ret_1d", "sector_ret_5d", "return_vs_sector",
        "sector_rel_momentum", "sector_encoded",
    ]
    missing = [f for f in mandatory if f not in df.columns]
    if missing:
        raise RuntimeError(f"Mandatory features missing: {missing}")
    log.info("All mandatory features present ✓")

    # ── 10. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)

    print(f"\n[OK] Saved: {MERGED_CSV}  shape={df.shape}")
    print(f"     Date range : {df['Date'].min()} -> {df['Date'].max()}")
    print(f"     Stocks     : {df['Stock'].nunique()}")
    print(f"     Labels  UP={up}({up/valid*100:.1f}%) "
          f"FLAT={flat}({flat/valid*100:.1f}%) DOWN={down}({down/valid*100:.1f}%)")
    print(f"     has_fundamental_data=1: {int(df['has_fundamental_data'].sum())} rows")
    if df["news_score"].abs().sum() > 0:
        print(f"     Real FinBERT news_score coverage: "
              f"{int(df['has_news'].sum())} rows ({int(df['has_news'].sum())/valid*100:.1f}%)")
    else:
        print("     WARNING: news_score is all 0 — run finbert/infer_news.py!")


if __name__ == "__main__":
    merge_all()
