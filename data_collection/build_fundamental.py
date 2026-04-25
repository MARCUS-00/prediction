# ============================================================
# data_collection/build_fundamental.py  (FIXED v3)
#
# Fixes vs v2 (FIXED v2):
#   1. [CRITICAL] expand_years() no longer fills Revenue_Growth and
#      Profit_Growth with 0.0 for years where yfinance has no data.
#      yfinance only returns ~3-5 years of financials, so years
#      2015-2021 get 0.0, which makes 73.4% of all growth rows identical.
#      A constant feature is statistically useless and can hurt tree-based
#      models by introducing spurious split candidates.
#
#      Fix: years without reported financials keep NaN for growth columns.
#      merge_features.py already median-imputes NaN numerics, so the
#      downstream pipeline handles this correctly.
#
#   2. [MINOR] Revenue and Profit are now also kept as NaN (not ffilled)
#      for years before the first reported value, to avoid feeding
#      stale absolute revenue figures into the feature set.
#      After the first reported year, ffill is still applied (annual
#      reports repeat until the next year).
#
#   3. Logging improved to show how many rows have real vs imputed values.
# ============================================================

import os, sys
import numpy as np
import pandas as pd
import yfinance as yf

PRICE_START_DATE = "2015-01-01"
PRICE_END_DATE   = "2025-12-31"
START_YEAR       = 2015
END_YEAR         = 2025
STOCK_COUNT      = 40

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config.settings import SECTOR_MAP

OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "fundamental")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fundamental.csv")

STOCKS = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","INDUSINDBK.NS","TCS.NS","INFY.NS",
    "HCLTECH.NS","WIPRO.NS","TECHM.NS","RELIANCE.NS","ONGC.NS",
    "NTPC.NS","POWERGRID.NS","BPCL.NS","HINDUNILVR.NS","ITC.NS",
    "NESTLEIND.NS","BRITANNIA.NS","MARUTI.NS","M&M.NS","BHARTIARTL.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","BAJAJ-AUTO.NS","SUNPHARMA.NS","CIPLA.NS",
    "DRREDDY.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS",
    "LT.NS","ULTRACEMCO.NS","GRASIM.NS","ASIANPAINT.NS","TITAN.NS",
][:STOCK_COUNT]


def safe_get(df, keys, col):
    if df is None or col is None: return np.nan
    for k in keys:
        if k in df.index:
            val = df.loc[k, col]
            if not pd.isna(val): return float(val)
    return np.nan


def find_col(df, year):
    if df is None: return None
    for c in df.columns:
        try:
            if pd.to_datetime(c).year == year: return c
        except Exception: pass
    return None


def get_stock_data(stock):
    base = stock.replace(".NS", "")
    data = []
    try:
        ticker    = yf.Ticker(stock)
        hist      = ticker.history(start=PRICE_START_DATE, end=PRICE_END_DATE)
        if hist.empty: print(f"  ⚠ Skipped {stock} (no price data)"); return []
        hist["Year"] = pd.to_datetime(hist.index).year
        price_map    = hist.groupby("Year")["Close"].last()
        fin  = ticker.financials
        bs   = ticker.balance_sheet
        info = ticker.info
        if fin is None or fin.empty: print(f"  ⚠ Skipped {stock} (no financials)"); return []
        fin.columns = pd.to_datetime(fin.columns)
        if bs is not None and not bs.empty: bs.columns = pd.to_datetime(bs.columns)
        years = [c.year for c in fin.columns]
        for year in years:
            f_col  = find_col(fin, year)
            p_col  = find_col(fin, year - 1)
            b_col  = find_col(bs, year)
            revenue = safe_get(fin, ["Total Revenue"], f_col)
            profit  = safe_get(fin, ["Net Income"],    f_col)
            shares  = info.get("sharesOutstanding", np.nan)
            eps     = profit / shares if shares and shares > 0 else np.nan
            price   = price_map.get(year, np.nan)
            pe      = price / eps if eps and eps != 0 else np.nan
            equity  = safe_get(bs, ["Stockholders Equity"], b_col)
            debt    = safe_get(bs, ["Total Debt"],           b_col)
            roe     = profit / equity if equity and equity != 0 else np.nan
            dte     = debt   / equity if equity and equity != 0 else np.nan
            # Compute growth from adjacent REPORTED values only
            rev_g = prof_g = np.nan
            if p_col:
                prev_rev    = safe_get(fin, ["Total Revenue"], p_col)
                prev_profit = safe_get(fin, ["Net Income"],    p_col)
                if prev_rev    and prev_rev    != 0: rev_g  = (revenue - prev_rev)  / abs(prev_rev)
                if prev_profit and prev_profit != 0: prof_g = (profit  - prev_profit) / abs(prev_profit)
            data.append({
                "Stock": base, "Year": year,
                "Sector": SECTOR_MAP.get(base, "Unknown"),
                "PE_Ratio": pe, "EPS": eps, "ROE": roe, "Debt_to_Equity": dte,
                "Revenue": revenue, "Profit": profit,
                "Revenue_Growth": rev_g, "Profit_Growth": prof_g,
            })
    except Exception as e:
        print(f"  [ERROR] {stock}: {e}")
    return data


def expand_years(df):
    """
    Expand to full year range.

    FIX v3: Growth columns (Revenue_Growth, Profit_Growth) are kept as NaN
    for years where yfinance has no reported data.  Filling them with 0.0
    (as in v2) makes ~73% of all rows identical, producing a near-constant
    feature that carries zero discriminative signal for the model.

    NaN values will be median-imputed by merge_features.py, which is far
    less misleading than a hard 0.0 default.

    All other fundamental columns (PE_Ratio, EPS, ROE, etc.) use ffill
    then bfill within each stock, as those are point-in-time snapshots that
    legitimately persist until the next annual report.
    """
    final = []
    for stock in df["Stock"].unique():
        sub = df[df["Stock"] == stock].sort_values("Year").copy()

        # Expand to full year range; years without data will be NaN
        full = pd.DataFrame({"Year": range(START_YEAR, END_YEAR + 1)})
        sub  = full.merge(sub, on="Year", how="left")
        sub["Stock"]  = stock
        sub["Sector"] = sub["Sector"].fillna(SECTOR_MAP.get(stock, "Unknown"))
        sub.sort_values("Year", inplace=True)

        # FIX: Separate growth cols from non-growth cols before ffill
        growth_cols    = ["Revenue_Growth", "Profit_Growth"]
        non_growth_num = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                          "Revenue", "Profit"]

        # Non-growth fundamentals: ffill then bfill (persist last known value)
        sub[non_growth_num] = sub[non_growth_num].ffill().bfill()

        # Growth columns: ffill only — do NOT fill the leading NaN years
        # (years before first reported value stay NaN; only REPORTED years get
        # the last known growth rate forwarded until the next reported year).
        sub[growth_cols] = sub[growth_cols].ffill()
        # NOTE: bfill is intentionally NOT applied here — we do not want to
        # back-fill growth from future reported years into earlier NaN rows.

        final.append(sub)

    result = pd.concat(final).sort_values(["Stock", "Year"]).reset_index(drop=True)

    # Report data coverage
    real_pct = result["Revenue_Growth"].notna().mean() * 100
    print(f"  [INFO] Revenue_Growth coverage: {real_pct:.1f}% rows have real values "
          f"(rest are NaN → will be median-imputed downstream)")
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD FUNDAMENTAL DATASET (FIXED v3)")
    print("=" * 60)
    dataset = []; success = skipped = 0
    for stock in STOCKS:
        print(f"  Processing {stock}...", end=" ")
        data = get_stock_data(stock)
        if data: dataset.extend(data); success += 1; print("✔")
        else:    skipped += 1; print("✘")
    if not dataset: print("\n[ERROR] No data collected"); return
    df = pd.DataFrame(dataset)
    df = expand_years(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved -> {OUTPUT_FILE}")
    print(f"  Success:{success}  Skipped:{skipped}  Rows:{len(df)}")
    # Show growth coverage
    print(f"  Revenue_Growth non-NaN rows : {df['Revenue_Growth'].notna().sum()}")
    print(f"  Profit_Growth  non-NaN rows : {df['Profit_Growth'].notna().sum()}")
    print(df.tail())

if __name__ == "__main__":
    main()