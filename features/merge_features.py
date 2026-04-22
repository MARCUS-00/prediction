# =============================================================================
# features/merge_features.py
# Merges technical, fundamental, news, and events CSVs into
# data/merged/merged_final.csv
# =============================================================================
import sys, os, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TECHNICAL_CSV, FUNDAMENTAL_CSV, EVENTS_CSV, MERGED_CSV,
    TECHNICAL_COLS, EVENTS_COLS, FUNDAMENTAL_FY_PREFIXES, FUNDAMENTAL_BASE_COLS,
    NEWS_CSV
)


def _p(tag, msg): print(f"  [{tag}] {msg}")

def _load_news():
    if not os.path.exists(NEWS_CSV): return pd.DataFrame()
    return pd.read_csv(NEWS_CSV)


# ── Step 1: Load & validate technical ────────────────────────────────────────
def _load_technical() -> pd.DataFrame:
    if not os.path.exists(TECHNICAL_CSV):
        _p("✗", "technical.csv not found. Run: python data_collection/build_technical.py")
        return pd.DataFrame()
    try:
        df = pd.read_csv(TECHNICAL_CSV)
        _p("✓", f"technical.csv  shape={df.shape}")
    except Exception as e:
        _p("✗", f"Cannot read technical.csv: {e}")
        return pd.DataFrame()

    missing = [c for c in TECHNICAL_COLS if c not in df.columns]
    if missing:
        _p("✗", f"technical.csv missing columns: {missing}")
        return pd.DataFrame()

    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)
    df.dropna(subset=["Date", "Close", "Direction"], inplace=True)
    df["Direction"] = df["Direction"].astype(int)
    df.drop_duplicates(subset=["Date", "Stock"], inplace=True)
    return df


# ── Step 2: Add lag + rolling features ───────────────────────────────────────
def _add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["Stock", "Date"], inplace=True)

    lag_map = {"Close": [1, 2, 3, 5], "RSI": [1], "MACD": [1], "OBV": [1], "Return_1d": [1]}
    added_lag = 0
    for col, lags in lag_map.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("Stock")[col].shift(lag)
            added_lag += 1

    added_roll = 0
    for w in [5, 10, 20]:
        df[f"Close_roll_mean_{w}"] = df.groupby("Stock")["Close"].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean())
        df[f"Close_roll_std_{w}"]  = df.groupby("Stock")["Close"].transform(
            lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0))
        added_roll += 2

    _p("✓", f"Lag features: {added_lag}  Rolling features: {added_roll}")
    return df


# ── Step 3: Load & rename fundamental ────────────────────────────────────────
def _load_fundamental() -> pd.DataFrame:
    if not os.path.exists(FUNDAMENTAL_CSV):
        _p("!", "fundamental.csv not found — fundamental cols will be NaN")
        return pd.DataFrame()
    try:
        df = pd.read_csv(FUNDAMENTAL_CSV)
        _p("✓", f"fundamental.csv  shape={df.shape}")
    except Exception as e:
        _p("!", f"Cannot read fundamental.csv: {e} — skipping")
        return pd.DataFrame()

    # Detect FY suffix dynamically e.g. PE_Ratio_FY24 → "24"
    fy = ""
    for col in df.columns:
        m = re.search(r"PE_Ratio_FY(\d+)", col)
        if m:
            fy = m.group(1)
            break

    if not fy:
        _p("!", f"Cannot detect FY suffix in fundamental.csv columns: {df.columns.tolist()}")
        return pd.DataFrame()

    _p("i", f"Detected FY suffix: FY{fy}")

    rename = {
        f"PE_Ratio_FY{fy}"      : "PE_Ratio",
        f"EPS_FY{fy}"           : "EPS",
        f"ROE_FY{fy}"           : "ROE",
        f"Debt_to_Equity_FY{fy}": "Debt_to_Equity",
    }
    actual = {k: v for k, v in rename.items() if k in df.columns}
    df.rename(columns=actual, inplace=True)
    _p("✓", f"Renamed FY columns: {list(actual.values())}")

    df["Stock"] = df["Stock"].astype(str).str.replace(".NS", "", regex=False)

    # Ensure all expected cols exist
    for col in ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity", "Revenue_Growth", "Profit_Growth"]:
        if col not in df.columns:
            df[col] = np.nan
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    # Clip outliers
    for col in ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity", "Revenue_Growth", "Profit_Growth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        q1, q99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q99)

    keep = ["Stock", "Sector", "PE_Ratio", "EPS", "ROE",
            "Debt_to_Equity", "Revenue_Growth", "Profit_Growth"]
    return df[[c for c in keep if c in df.columns]]


# ── Step 4: Load & normalise events ──────────────────────────────────────────
def _load_events() -> pd.DataFrame:
    if not os.path.exists(EVENTS_CSV):
        _p("!", "events.csv not found — event cols will be 0")
        return pd.DataFrame()
    try:
        df = pd.read_csv(EVENTS_CSV)
        _p("✓", f"events.csv  shape={df.shape}")
    except Exception as e:
        _p("!", f"Cannot read events.csv: {e} — skipping")
        return pd.DataFrame()

    missing = [c for c in EVENTS_COLS if c not in df.columns]
    if missing:
        _p("!", f"events.csv missing columns: {missing} — skipping")
        return pd.DataFrame()

    # Normalise lowercase → Title case for merging
    df.rename(columns={"date": "Date", "symbol": "Stock"}, inplace=True)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["Stock"] = df["Stock"].astype(str).str.strip()
    df.dropna(subset=["Date", "Stock"], inplace=True)

    for col in ["event_score_max", "event_count", "is_event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["event_category", "event_name"]:
        df[col] = df[col].fillna("NONE")

    _p("✓", f"Events normalised  shape={df.shape}")
    return df


# ── Main merge ────────────────────────────────────────────────────────────────
def merge_all() -> pd.DataFrame:
    print("\n" + "="*55)
    print("  MERGE FEATURES → merged_final.csv")
    print("="*55)

    # 1. Technical base
    tech = _load_technical()
    if tech.empty:
        _p("✗", "Cannot proceed without technical.csv")
        return pd.DataFrame()
    tech   = _add_lag_rolling(tech)
    merged = tech.copy()
    _p("i", f"Base rows: {len(merged)}")

    # 2. Fundamental (one row per Stock — broadcast)
    fund = _load_fundamental()
    if not fund.empty:
        merged = merged.merge(fund, on="Stock", how="left")
        _p("✓", f"After fundamental merge: {merged.shape}")
        
        # Null out fundamentals for historical rows where FY25 data is anachronistic
        merged.loc[merged["Date"] < "2024-01-01", 
                   ["PE_Ratio","EPS","ROE","Debt_to_Equity","Revenue_Growth","Profit_Growth"]] = np.nan
    else:
        for col in ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                    "Revenue_Growth", "Profit_Growth"]:
            merged[col] = np.nan
        merged["Sector"] = "Unknown"

    # 3. News (average per Date+Stock)
    sent = _load_news()
    if not sent.empty:
        s_cols = [c for c in ["Date", "Stock", "news_positive",
                               "news_neutral", "news_negative"]
                  if c in sent.columns]
        sent["Stock"] = sent["Stock"].astype(str).str.replace(".NS", "", regex=False)
        daily  = sent[s_cols].groupby(["Date", "Stock"], as_index=False).mean(numeric_only=True)
        merged = merged.merge(daily, on=["Date", "Stock"], how="left")
        _p("✓", f"After news merge: {merged.shape}")
    else:
        _p("!", "News unavailable — filling neutral 0.333")

    for col, fill in [("news_positive", 0.333),
                      ("news_neutral",  0.334),
                      ("news_negative", 0.333)]:
        if col not in merged.columns:
            merged[col] = fill
        else:
            merged[col] = merged[col].fillna(fill)

    # 4. Events (on Date+Stock)
    ev = _load_events()
    if not ev.empty:
        ev_cols = ["Date", "Stock", "event_category", "event_name",
                   "event_score_max", "event_count", "is_event"]
        ev_cols = [c for c in ev_cols if c in ev.columns]
        merged  = merged.merge(ev[ev_cols], on=["Date", "Stock"], how="left")
        _p("✓", f"After events merge: {merged.shape}")
    else:
        _p("!", "Events unavailable — filling zeros")

    for col, fill in [("event_score_max", 0.0), ("event_count", 0), ("is_event", 0)]:
        if col not in merged.columns:
            merged[col] = fill
        else:
            merged[col] = merged[col].fillna(fill)
    for col in ["event_category", "event_name"]:
        if col not in merged.columns:
            merged[col] = "NONE"
        else:
            merged[col] = merged[col].fillna("NONE")

    # 5. Final clean
    before = len(merged)
    merged.dropna(subset=["Close", "Direction", "RSI", "MACD"], inplace=True)
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    _p("i", f"Dropped {before - len(merged)} rows (null Close/Direction/RSI/MACD)")

    merged.sort_values(["Date", "Stock"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # 6. Save
    try:
        os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
        merged.to_csv(MERGED_CSV, index=False)
        _p("✓", f"Saved → {MERGED_CSV}")
    except IOError as e:
        _p("✗", f"Cannot save merged_final.csv: {e}")

    print("="*55)
    _p("✓", f"DONE  shape={merged.shape}")
    _p("✓", f"Stocks : {sorted(merged['Stock'].unique().tolist())}")
    _p("✓", f"Dates  : {merged['Date'].min()} → {merged['Date'].max()}")
    _p("✓", f"Columns: {len(merged.columns)}")
    print("="*55 + "\n")
    return merged


if __name__ == "__main__":
    merge_all()
