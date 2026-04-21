# ============================================================
# build_fundamental.py
# Fetches point-in-time fundamental data for NIFTY 50 stocks.
# Outputs: data/fundamental/fundamental.csv
# Install: pip install yfinance pandas numpy
# ============================================================

import os, sys
import numpy as np
import pandas as pd
import yfinance as yf
from contextlib import contextmanager

START_DATE  = "2015-01-01"
END_DATE    = "2025-12-31"
STOCK_COUNT = 40

_end_dt     = pd.to_datetime(END_DATE)
TARGET_YEAR = _end_dt.year
PREV_YEAR   = TARGET_YEAR - 1

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "fundamental")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fundamental.csv")

STOCKS = [
    "HDFCBANK.NS",  "ICICIBANK.NS",  "SBIN.NS",       "AXISBANK.NS",  "KOTAKBANK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS", "INDUSINDBK.NS", "TCS.NS",       "INFY.NS",
    "HCLTECH.NS",   "WIPRO.NS",      "TECHM.NS",      "RELIANCE.NS",  "ONGC.NS",
    "NTPC.NS",      "POWERGRID.NS",  "BPCL.NS",       "HINDUNILVR.NS","ITC.NS",
    "NESTLEIND.NS", "BRITANNIA.NS",  "MARUTI.NS",     "M&M.NS",       "BHARTIARTL.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "SUNPHARMA.NS", "CIPLA.NS",
    "DRREDDY.NS",   "TATASTEEL.NS",  "JSWSTEEL.NS",   "HINDALCO.NS",  "COALINDIA.NS",
    "LT.NS",        "ULTRACEMCO.NS", "GRASIM.NS",     "ASIANPAINT.NS","TITAN.NS",
][:STOCK_COUNT]

SECTOR_MAP = {
    "HDFCBANK":   "Financial Services", "ICICIBANK":  "Financial Services",
    "SBIN":       "Financial Services", "AXISBANK":   "Financial Services",
    "KOTAKBANK":  "Financial Services", "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services", "INDUSINDBK": "Financial Services",
    "TCS":        "Technology",         "INFY":       "Technology",
    "HCLTECH":    "Technology",         "WIPRO":      "Technology",
    "TECHM":      "Technology",         "RELIANCE":   "Energy",
    "ONGC":       "Energy",             "NTPC":       "Utilities",
    "POWERGRID":  "Utilities",          "BPCL":       "Energy",
    "COALINDIA":  "Energy",             "HINDUNILVR": "Consumer Defensive",
    "ITC":        "Consumer Defensive", "NESTLEIND":  "Consumer Defensive",
    "BRITANNIA":  "Consumer Defensive", "MARUTI":     "Consumer Cyclical",
    "M&M":        "Consumer Cyclical",  "BHARTIARTL": "Communication Services",
    "EICHERMOT":  "Consumer Cyclical",  "HEROMOTOCO": "Consumer Cyclical",
    "BAJAJ-AUTO": "Consumer Cyclical",  "TITAN":      "Consumer Cyclical",
    "SUNPHARMA":  "Healthcare",         "CIPLA":      "Healthcare",
    "DRREDDY":    "Healthcare",         "TATASTEEL":  "Basic Materials",
    "JSWSTEEL":   "Basic Materials",    "HINDALCO":   "Basic Materials",
    "ULTRACEMCO": "Basic Materials",    "GRASIM":     "Basic Materials",
    "ASIANPAINT": "Basic Materials",    "LT":         "Industrials",
}

SYMBOL_FALLBACKS = {"M&M.NS": ["M&M.NS", "MM.NS"]}


@contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _safe_get(df: pd.DataFrame, row_names: list, col) -> float:
    for name in row_names:
        if name in df.index:
            try:
                val = df.loc[name, col]
                if not pd.isna(val):
                    return float(val)
            except Exception:
                pass
    return np.nan


def _find_col(df: pd.DataFrame, year: int):
    for c in df.columns:
        try:
            if pd.to_datetime(c).year == year:
                return c
        except Exception:
            pass
    return None


def get_stock_data(base_ticker: str) -> dict:
    symbols     = SYMBOL_FALLBACKS.get(base_ticker, [base_ticker])
    ticker_base = base_ticker.replace(".NS", "")
    fy          = str(TARGET_YEAR)[-2:]

    result = {
        "Stock"                  : ticker_base,
        "Sector"                 : SECTOR_MAP.get(ticker_base, "Unknown"),
        f"PE_Ratio_FY{fy}"       : np.nan,
        f"EPS_FY{fy}"            : np.nan,
        f"ROE_FY{fy}"            : np.nan,
        f"Debt_to_Equity_FY{fy}" : np.nan,
        "Revenue_Growth"         : np.nan,
        "Profit_Growth"          : np.nan,
    }

    stock = hist = info = None
    with _suppress_output():
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                h = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
                i = t.info
                if not h.empty and i:
                    stock, hist, info = t, h, i
                    break
            except Exception:
                continue

    if stock is None or hist is None or hist.empty or not info:
        return result

    fin = bs = c_target = c_prev = b_target = None
    try:
        fin = stock.financials
        if fin is not None and not fin.empty:
            fin.columns = pd.to_datetime(fin.columns, errors="coerce")
            c_target    = _find_col(fin, TARGET_YEAR)
            c_prev      = _find_col(fin, PREV_YEAR)
        bs = stock.balance_sheet
        if bs is not None and not bs.empty:
            bs.columns = pd.to_datetime(bs.columns, errors="coerce")
            b_target   = _find_col(bs, TARGET_YEAR)
    except Exception:
        pass

    # EPS
    eps = np.nan
    if c_target is not None and fin is not None:
        try:
            net_inc = _safe_get(fin, ["Net Income"], c_target)
            shares  = info.get("sharesOutstanding")
            if shares and shares > 0 and not pd.isna(net_inc):
                eps = net_inc / shares
                if (info.get("currency") == "INR"
                        and info.get("financialCurrency") == "USD"
                        and abs(eps) < 10):
                    eps = eps * 83.0
                result[f"EPS_FY{fy}"] = round(float(eps), 4)
        except Exception:
            pass

    # P/E
    try:
        last_close = float(hist["Close"].iloc[-1])
        if not pd.isna(eps) and eps > 0:
            result[f"PE_Ratio_FY{fy}"] = round(last_close / eps, 4)
    except Exception:
        pass

    # Revenue & Profit Growth
    if c_target is not None and c_prev is not None and fin is not None:
        try:
            rev_t = _safe_get(fin, ["Total Revenue"], c_target)
            rev_p = _safe_get(fin, ["Total Revenue"], c_prev)
            if not pd.isna(rev_t) and not pd.isna(rev_p) and rev_p != 0:
                result["Revenue_Growth"] = round((rev_t - rev_p) / abs(rev_p), 4)
        except Exception:
            pass
        try:
            net_t = _safe_get(fin, ["Net Income"], c_target)
            net_p = _safe_get(fin, ["Net Income"], c_prev)
            if not pd.isna(net_t) and not pd.isna(net_p) and net_p != 0:
                result["Profit_Growth"] = round((net_t - net_p) / abs(net_p), 4)
        except Exception:
            pass

    # Debt/Equity & ROE
    if b_target is not None and bs is not None:
        try:
            total_debt = _safe_get(bs, ["Total Debt"], b_target)
            if pd.isna(total_debt): total_debt = 0.0
            cash = _safe_get(bs, [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
            ], b_target)
            if pd.isna(cash): cash = 0.0
            equity = _safe_get(bs, ["Stockholders Equity", "Common Stock Equity"], b_target)
            if not pd.isna(equity) and equity != 0:
                result[f"Debt_to_Equity_FY{fy}"] = round((total_debt - cash) / equity, 4)
                if c_target is not None and fin is not None:
                    net_t = _safe_get(fin, ["Net Income"], c_target)
                    if not pd.isna(net_t):
                        result[f"ROE_FY{fy}"] = round(net_t / equity, 4)
        except Exception:
            pass

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"  BUILD FUNDAMENTAL DATASET")
    print(f"  Date range   : {START_DATE} → {END_DATE}")
    print(f"  Target FY    : FY{str(TARGET_YEAR)[-2:]} ({TARGET_YEAR})")
    print(f"  Stocks       : {len(STOCKS)}")
    print(f"  Output       : {OUTPUT_FILE}")
    print("=" * 60)

    dataset = []
    success = skipped = 0

    for ticker in STOCKS:
        print(f"  {ticker:<22}", end="", flush=True)
        try:
            data = get_stock_data(ticker)
            if data:
                dataset.append(data)
                success += 1
                print("[OK]")
            else:
                skipped += 1
                print("[SKIP] No data")
        except Exception as e:
            skipped += 1
            print(f"[ERROR] {e}")

    if not dataset:
        print("\n[ERROR] No fundamental data collected.")
        return

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_FILE, index=False)

    pd.options.display.float_format = "{:.2f}".format
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)

    print("\n" + "=" * 60)
    print(f"  ✅ Saved → {OUTPUT_FILE}")
    print(f"     Stocks : {len(df)}  (success={success}, skip={skipped})")
    print("=" * 60)


if __name__ == "__main__":
    main()
