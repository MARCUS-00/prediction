import os, sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SECTOR_MAP, DATE_START, DATE_END

PRICE_START_DATE = DATE_START
PRICE_END_DATE   = DATE_END
START_YEAR       = int(DATE_START[:4])
END_YEAR         = int(DATE_END[:4])
STOCK_COUNT      = 40

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    if df is None or col is None:
        return np.nan
    for k in keys:
        if k in df.index:
            val = df.loc[k, col]
            if not pd.isna(val):
                return float(val)
    return np.nan


def find_col(df, year):
    if df is None:
        return None
    for c in df.columns:
        try:
            if pd.to_datetime(c).year == year:
                return c
        except Exception:
            pass
    return None


def get_stock_data(stock):
    base = stock.replace(".NS", "")
    data = []
    try:
        ticker = yf.Ticker(stock)
        hist = ticker.history(start=PRICE_START_DATE, end=PRICE_END_DATE)
        if hist.empty:
            print(f"  [SKIP] {stock} (no price data)")
            return []
        hist["Year"] = pd.to_datetime(hist.index).year
        price_map    = hist.groupby("Year")["Close"].last()
        fin  = ticker.financials
        bs   = ticker.balance_sheet
        info = ticker.info
        if fin is None or fin.empty:
            print(f"  [SKIP] {stock} (no financials)")
            return []
        fin.columns = pd.to_datetime(fin.columns)
        if bs is not None and not bs.empty:
            bs.columns = pd.to_datetime(bs.columns)
        years = [c.year for c in fin.columns]
        for year in years:
            if year < START_YEAR or year > END_YEAR:
                continue
            f_col = find_col(fin, year)
            p_col = find_col(fin, year - 1)
            b_col = find_col(bs,  year)
            revenue = safe_get(fin, ["Total Revenue"], f_col)
            profit  = safe_get(fin, ["Net Income"],    f_col)
            shares  = info.get("sharesOutstanding", np.nan)
            eps     = profit / shares if shares and shares > 0 else np.nan
            price   = price_map.get(year, np.nan)
            pe      = price / eps if eps and eps != 0 else np.nan
            equity  = safe_get(bs, ["Stockholders Equity"], b_col)
            debt    = safe_get(bs, ["Total Debt"],          b_col)
            roe     = profit / equity if equity and equity != 0 else np.nan
            dte     = debt   / equity if equity and equity != 0 else np.nan
            rev_g = prof_g = np.nan
            if p_col:
                prev_rev    = safe_get(fin, ["Total Revenue"], p_col)
                prev_profit = safe_get(fin, ["Net Income"],    p_col)
                if prev_rev    and prev_rev    != 0:
                    rev_g  = (revenue - prev_rev)    / abs(prev_rev)
                if prev_profit and prev_profit != 0:
                    prof_g = (profit  - prev_profit) / abs(prev_profit)
            data.append({
                "Stock":           base,
                "Year":            year,
                "Sector":          SECTOR_MAP.get(base, "Unknown"),
                "PE_Ratio":        pe,
                "EPS":             eps,
                "ROE":             roe,
                "Debt_to_Equity":  dte,
                "Revenue":         revenue,
                "Profit":          profit,
                "Revenue_Growth":  rev_g,
                "Profit_Growth":   prof_g,
            })
    except Exception as e:
        print(f"  [ERROR] {stock}: {e}")
    return data


def expand_years(df):
    """
    BUG-2 FIX: Do NOT back-fill or freeze values across years.

    Old behaviour: ffill().bfill() on PE_Ratio/ROE/EPS propagated the most
    recent year's value into ALL prior years (2020-2022), making all three years
    look identical. The model learned fundamentals never change, then was shocked
    by real variation in 2023-2025.

    New behaviour:
      - Only forward-fill (ffill) within a stock's own history — never backward.
      - Revenue_Growth and Profit_Growth for the FIRST year (no prior year to
        compare) remain NaN — they will be imputed in merge_features.py using
        sector-year median, which is far more accurate than frozen values.
      - PE_Ratio / ROE / EPS are NOT back-filled to years before data exists;
        NaN is preserved so merge_features.py imputation handles them properly.
    """
    final = []
    for stock in df["Stock"].unique():
        sub  = df[df["Stock"] == stock].sort_values("Year").copy()
        full = pd.DataFrame({"Year": range(START_YEAR, END_YEAR + 1)})
        sub  = full.merge(sub, on="Year", how="left")
        sub["Stock"]  = stock
        sub["Sector"] = sub["Sector"].fillna(SECTOR_MAP.get(stock, "Unknown"))
        sub.sort_values("Year", inplace=True)

        # Forward-fill only: carry known values forward (e.g., 2023 data into
        # 2024 if 2024 filing not yet available). Do NOT backward-fill.
        fwd_fill_cols = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                         "Revenue", "Profit"]
        # Revenue_Growth / Profit_Growth are NOT forward-filled (each year
        # has its own real growth rate; filling would repeat stale rates).
        sub[fwd_fill_cols] = sub[fwd_fill_cols].ffill()
        # Leave Revenue_Growth, Profit_Growth as NaN where not available —
        # merge_features.py will fill with sector-year median.

        final.append(sub)

    result = pd.concat(final).sort_values(["Stock", "Year"]).reset_index(drop=True)
    real_pct = result["Revenue_Growth"].notna().mean() * 100
    nan_pct  = result["Revenue_Growth"].isna().mean() * 100
    print(f"  [INFO] BUG-2 FIX: Revenue_Growth — "
          f"{real_pct:.1f}% real values, {nan_pct:.1f}% NaN "
          f"(will be imputed by sector-year median in merge_features.py)")
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD FUNDAMENTAL DATASET")
    print(f"  Year range : {START_YEAR} -> {END_YEAR}")
    print("=" * 60)

    dataset = []
    success = skipped = 0
    for stock in STOCKS:
        print(f"  Processing {stock}...", end=" ")
        data = get_stock_data(stock)
        if data:
            dataset.extend(data)
            success += 1
            print("OK")
        else:
            skipped += 1
            print("FAIL")

    if not dataset:
        print("\n[ERROR] No data collected")
        return
    df = pd.DataFrame(dataset)
    df = expand_years(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved -> {OUTPUT_FILE}")
    print(f"  Success:{success}  Skipped:{skipped}  Rows:{len(df)}")
    print(f"  Year range in output: {df['Year'].min()} -> {df['Year'].max()}")


if __name__ == "__main__":
    main()
