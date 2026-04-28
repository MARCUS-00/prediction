# ============================================================
# data_collection/build_technical.py  (FIXED v12)
#
# FIX: START_DATE enforced to 2020-01-01 as required.
#   Old code used 2015-01-01 — caused the pre-2020 leak into
#   merged_final.csv. The merge_features.py also filters, but
#   starting clean here saves disk space and is correct by design.
#
# FIX: PREDICTION_HORIZON removed from label computation.
#   The 'Direction' column computed here is a raw indicator only.
#   The actual training target (alpha_5d) is computed in merge_features.py.
#   Keeping Direction = next-day sign for reference only.
# ============================================================

import os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from contextlib import contextmanager

START_DATE  = "2020-01-01"   # FIX: was 2015-01-01
END_DATE    = "2025-12-31"
STOCK_COUNT = 40

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "technical")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "technical.csv")

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

SYMBOL_FALLBACKS = {"M&M.NS": ["M&M.NS", "MM.NS"]}

OUTPUT_COLUMNS = [
    "Date", "Stock", "Open", "High", "Low", "Close", "Volume",
    "EMA_20", "RSI", "MACD", "MACD_signal", "ATR", "OBV",
    "Return_1d", "Direction",
]


@contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def fetch_and_process(base_symbol):
    symbols = SYMBOL_FALLBACKS.get(base_symbol, [base_symbol])
    df = pd.DataFrame()
    with _suppress_output():
        for sym in symbols:
            try:
                temp = yf.download(sym, start=START_DATE, end=END_DATE,
                                   interval="1d", progress=False, auto_adjust=True)
                if not temp.empty:
                    df = temp
                    break
            except Exception:
                continue

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df.dropna(subset=["Close"], inplace=True)

    if len(df) < 60:
        return pd.DataFrame()

    df["Stock"] = base_symbol.replace(".NS", "")

    try:
        df["EMA_20"]      = ta.trend.EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df["RSI"]         = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
        macd              = ta.trend.MACD(close=df["Close"])
        df["MACD"]        = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["ATR"]         = ta.volatility.AverageTrueRange(
                                high=df["High"], low=df["Low"], close=df["Close"]
                            ).average_true_range()
        df["OBV"]         = ta.volume.OnBalanceVolumeIndicator(
                                close=df["Close"], volume=df["Volume"]
                            ).on_balance_volume()
    except Exception as e:
        print(f"\n  [!] Indicator error for {base_symbol}: {e}")
        return pd.DataFrame()

    df["Return_1d"] = df["Close"].pct_change()

    # Direction = sign of next-day return (reference only; real target is alpha_5d in merge)
    df["Direction"] = np.where(df["Close"].pct_change(1).shift(-1) > 0, 1, -1)

    df.reset_index(inplace=True)
    if df.columns[0] != "Date":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Enforce date range after reset
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD TECHNICAL DATASET (FIXED v12)")
    print(f"  Date range : {START_DATE} -> {END_DATE}")
    print(f"  Stocks     : {len(STOCKS)}")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 60)

    all_data = []
    success = skipped = 0

    for symbol in STOCKS:
        print(f"  {symbol:<22}", end="", flush=True)
        try:
            df = fetch_and_process(symbol)
            if df is not None and not df.empty:
                all_data.append(df)
                success += 1
                print(f"[OK]  {len(df)} rows")
            else:
                skipped += 1
                print("[SKIP] No data / insufficient rows")
        except Exception as e:
            skipped += 1
            print(f"[ERROR] {e}")
        time.sleep(1)

    if not all_data:
        print("\n[ERROR] No data downloaded.")
        return

    final = pd.concat(all_data, ignore_index=True)
    final.sort_values(by=["Date", "Stock"], inplace=True)
    final.dropna(subset=["Close", "EMA_20", "RSI", "MACD", "ATR", "OBV"],
                 inplace=True)
    final.reset_index(drop=True, inplace=True)

    cols  = [c for c in OUTPUT_COLUMNS if c in final.columns]
    final = final[cols]
    final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"  [OK] Saved -> {OUTPUT_FILE}")
    print(f"     Shape      : {final.shape}")
    print(f"     Date range : {final['Date'].min()} -> {final['Date'].max()}")
    print(f"     Stocks     : {final['Stock'].nunique()}  "
          f"(success={success}, skip={skipped})")
    pre2020 = (final["Date"] < "2020-01-01").sum()
    print(f"     Pre-2020 rows: {pre2020}  (should be 0)")
    print("=" * 60)


if __name__ == "__main__":
    main()