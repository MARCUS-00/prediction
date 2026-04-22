# ============================================================
# build_fundamental.py
# Multi-year fundamental dataset (FINAL CLEAN VERSION)
# ============================================================

import os
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================
PRICE_START_DATE = "2015-01-01"
PRICE_END_DATE   = "2025-12-31"
START_YEAR       = 2015
END_YEAR         = 2025
STOCK_COUNT      = 40

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "fundamental")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fundamental.csv")

STOCKS = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","INDUSINDBK.NS","TCS.NS","INFY.NS",
    "HCLTECH.NS","WIPRO.NS","TECHM.NS","RELIANCE.NS","ONGC.NS",
    "NTPC.NS","POWERGRID.NS","BPCL.NS","HINDUNILVR.NS","ITC.NS",
    "NESTLEIND.NS","BRITANNIA.NS","MARUTI.NS","M&M.NS","BHARTIARTL.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","BAJAJ-AUTO.NS","SUNPHARMA.NS","CIPLA.NS",
    "DRREDDY.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS",
    "LT.NS","ULTRACEMCO.NS","GRASIM.NS","ASIANPAINT.NS","TITAN.NS"
][:STOCK_COUNT]

# =========================
# HELPERS
# =========================
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
        except:
            pass
    return None

# =========================
# FETCH DATA
# =========================
def get_stock_data(stock):
    base = stock.replace(".NS","")
    data = []

    try:
        ticker = yf.Ticker(stock)

        hist = ticker.history(start=PRICE_START_DATE, end=PRICE_END_DATE)
        if hist.empty:
            print(f"  ⚠ Skipped {stock} (no price data)")
            return []

        hist["Year"] = pd.to_datetime(hist.index).year
        price_map = hist.groupby("Year")["Close"].last()

        fin = ticker.financials
        bs  = ticker.balance_sheet
        info = ticker.info

        if fin is None or fin.empty:
            print(f"  ⚠ Skipped {stock} (no financials)")
            return []

        fin.columns = pd.to_datetime(fin.columns)
        if bs is not None and not bs.empty:
            bs.columns = pd.to_datetime(bs.columns)

        years = [c.year for c in fin.columns]

        for year in years:
            f_col = find_col(fin, year)
            p_col = find_col(fin, year-1)
            b_col = find_col(bs, year)

            revenue = safe_get(fin, ["Total Revenue"], f_col)
            profit  = safe_get(fin, ["Net Income"], f_col)

            shares = info.get("sharesOutstanding", np.nan)
            eps = profit / shares if shares else np.nan

            price = price_map.get(year, np.nan)
            pe = price/eps if eps and eps != 0 else np.nan

            equity = safe_get(bs, ["Stockholders Equity"], b_col)
            debt   = safe_get(bs, ["Total Debt"], b_col)

            roe = profit/equity if equity else np.nan
            dte = debt/equity if equity else np.nan

            rev_g = np.nan
            prof_g = np.nan

            if p_col:
                prev_rev = safe_get(fin, ["Total Revenue"], p_col)
                prev_profit = safe_get(fin, ["Net Income"], p_col)

                if prev_rev and prev_rev != 0:
                    rev_g = (revenue-prev_rev)/abs(prev_rev)

                if prev_profit and prev_profit != 0:
                    prof_g = (profit-prev_profit)/abs(prev_profit)

            data.append({
                "Stock": base,
                "Year": year,
                "PE_Ratio": pe,
                "EPS": eps,
                "ROE": roe,
                "Debt_to_Equity": dte,
                "Revenue": revenue,
                "Profit": profit,
                "Revenue_Growth": rev_g,
                "Profit_Growth": prof_g
            })

    except Exception as e:
        print(f"  ❌ Error {stock}: {e}")

    return data

# =========================
# EXPAND YEARS
# =========================
def expand_years(df):
    final = []

    for stock in df["Stock"].unique():
        sub = df[df["Stock"] == stock].copy()

        full = pd.DataFrame({"Year": range(START_YEAR, END_YEAR+1)})
        sub = full.merge(sub, on="Year", how="left")

        sub["Stock"] = stock
        sub = sub.sort_values("Year")

        sub = sub.ffill()
        sub = sub.bfill()

        final.append(sub)

    df = pd.concat(final).sort_values(["Stock","Year"])

    # ✅ recompute growth properly
    df["Revenue_Growth"] = df.groupby("Stock")["Revenue"].pct_change()
    df["Profit_Growth"] = df.groupby("Stock")["Profit"].pct_change()

    return df

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("  BUILD FUNDAMENTAL DATASET (FINAL)")
    print("="*60)

    dataset = []
    success = 0
    skipped = 0

    for stock in STOCKS:
        print(f"Processing {stock}...", end=" ")

        data = get_stock_data(stock)

        if data:
            dataset.extend(data)
            success += 1
            print("✔")
        else:
            skipped += 1
            print("✘")

    if not dataset:
        print("\n❌ No data collected")
        return

    df = pd.DataFrame(dataset)

    # Expand years
    df = expand_years(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*60)
    print(f"✅ Saved → {OUTPUT_FILE}")
    print(f"✔ Success: {success} stocks")
    print(f"✘ Skipped: {skipped} stocks")
    print(f"📊 Total rows: {len(df)}")
    print("="*60)

    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()