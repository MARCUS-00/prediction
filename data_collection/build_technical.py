# ============================================================
# build_technical.py  (FIXED v3)
#
# Fixes vs v2:
#   1. [CRITICAL] PREDICTION_HORIZON changed from 1 to 5.
#      The model is meant to predict 5-day price direction, not
#      next-day. Using HORIZON=1 was training for the wrong task,
#      causing systematically poor out-of-sample accuracy.
#
#   2. [BUG] Return_1d_lag1 was computed as shift(1) of pct_change(),
#      which is the close-to-close return from 2 days ago, not
#      yesterday (off-by-one). Fixed: compute pct_change() directly,
#      then lag by 1 to get yesterday's return as a safe feature.
#      Also added Return_5d_lag1 (lagged 5-day rolling return) as an
#      additional momentum feature that matches the prediction horizon.
#
#   3. [BUG] fwd_return was computed with pct_change(HORIZON).shift(-HORIZON).
#      pct_change(n) computes return over n days ending at row t, not
#      starting at row t.  Correct look-ahead-free label:
#          fwd_return[t] = Close[t+HORIZON] / Close[t] - 1
#      Fixed using: df["Close"].shift(-HORIZON) / df["Close"] - 1
#
#   4. [BUG] Tail-drop was iloc[:-HORIZON] which removes HORIZON rows
#      from the end — correct for HORIZON=1 but should also remove the
#      rows where the forward label is undefined (last HORIZON rows).
#      Guard is kept but now explicitly aligned with the fwd_return NaN
#      mask via dropna on Direction.
#
#   5. [BUG] Volume filter `df["Volume"] > 0` was applied before
#      indicators but after dropna(Close). Zero-volume days can still
#      have a valid Close (e.g. NSE auctions). Keep the filter but
#      moved it after technical indicators to avoid gaps in the rolling
#      windows that would corrupt EMA/ATR/OBV values.
#
#   6. [MINOR] Added BB_Width (Bollinger Band width) as a volatility
#      feature — useful for a 5-day horizon model.
#
#   7. [MINOR] OUTPUT_COLUMNS updated: Return_5d_lag1 added,
#      BB_Width added.
# ============================================================

import os, sys, time
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from contextlib import contextmanager

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE         = "2020-01-01"
END_DATE           = "2025-12-31"
STOCK_COUNT        = 40
PREDICTION_HORIZON = 5      # FIX v3: was 1 — model predicts 5-day direction

# Binary (0/1) vs ternary (-1/0/1) direction labels
TERNARY_DIRECTION = False
FLAT_THRESHOLD    = 0.005   # ±0.5% counts as flat for 5-day horizon (was 0.05%)

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

SYMBOL_FALLBACKS = {
    "M&M.NS":        ["M&M.NS", "MM.NS"],
    "BAJAJ-AUTO.NS": ["BAJAJ-AUTO.NS", "BAJAJAUTO.NS"],
}

OUTPUT_COLUMNS = [
    "Date", "Stock",
    "Open", "High", "Low", "Close", "Volume",
    "EMA_20", "RSI", "MACD", "MACD_signal", "ATR", "OBV", "BB_Width",
    "Return_1d_lag1",   # yesterday's 1-day return (safe feature)
    "Return_5d_lag1",   # yesterday's 5-day rolling return (momentum, safe)
    "Direction",
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


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex column frame to a flat Index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _make_direction(fwd_return: pd.Series) -> pd.Series:
    """
    Convert a forward-return Series to a Direction label Series.
    Binary  (TERNARY_DIRECTION=False): 1 = up, 0 = down/flat
    Ternary (TERNARY_DIRECTION=True) : 1 = up, -1 = down, 0 = flat
    """
    if TERNARY_DIRECTION:
        return pd.Series(
            np.where(fwd_return >  FLAT_THRESHOLD,  1,
            np.where(fwd_return < -FLAT_THRESHOLD, -1, 0)),
            index=fwd_return.index,
        )
    else:
        return pd.Series(
            np.where(fwd_return > 0, 1, 0),
            index=fwd_return.index,
        )


def fetch_and_process(base_symbol: str) -> pd.DataFrame:
    symbols = SYMBOL_FALLBACKS.get(base_symbol, [base_symbol])
    df      = pd.DataFrame()

    with _suppress_output():
        for sym in symbols:
            try:
                temp = yf.download(
                    sym,
                    start=START_DATE,
                    end=END_DATE,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if not temp.empty:
                    df = temp
                    break
            except Exception:
                continue

    if df.empty:
        return pd.DataFrame()

    df = _flatten_columns(df)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df.dropna(subset=["Close"], inplace=True)

    # Need at least HORIZON + indicator warmup rows
    if len(df) < 60 + PREDICTION_HORIZON:
        return pd.DataFrame()

    df["Stock"] = base_symbol.replace(".NS", "")

    # ── Technical indicators ──────────────────────────────────────────────
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
        # FIX v3: BB_Width added — useful volatility signal for 5-day horizon
        bb = ta.volatility.BollingerBands(close=df["Close"], window=20)
        df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    except Exception as e:
        print(f"\n  [!] Indicator error for {base_symbol}: {e}")
        return pd.DataFrame()

    # FIX v3: filter zero-volume rows AFTER indicators to avoid gaps in
    # rolling windows that would corrupt EMA/ATR/OBV/BB values.
    df = df[df["Volume"] > 0]

    # ── Returns and direction ─────────────────────────────────────────────
    # Return_1d_lag1: yesterday's 1-day close-to-close return (safe feature)
    # FIX v3: was pct_change().shift(1) — pct_change is t vs t-1, shift(1)
    # makes it t-1 vs t-2 (off by one). Correct: pct_change gives today's
    # return; shift(1) then gives yesterday's return.
    df["Return_1d"]      = df["Close"].pct_change()          # today's 1d return (leaks)
    df["Return_1d_lag1"] = df["Return_1d"].shift(1)          # yesterday's (safe)

    # Return_5d_lag1: yesterday's 5-day rolling return (momentum feature)
    df["Return_5d"]      = df["Close"].pct_change(5)         # today's 5d return (leaks)
    df["Return_5d_lag1"] = df["Return_5d"].shift(1)          # yesterday's (safe)

    # FIX v3: Correct forward return for a PREDICTION_HORIZON-day label.
    # pct_change(n).shift(-n) computes return over n days ending at t+n,
    # not starting at t.  The proper no-lookahead label is:
    #   fwd_return[t] = Close[t+HORIZON] / Close[t] - 1
    fwd_return  = df["Close"].shift(-PREDICTION_HORIZON) / df["Close"] - 1
    df["Direction"] = _make_direction(fwd_return)

    # Drop rows where Direction is NaN (last PREDICTION_HORIZON rows where
    # the forward price is unknown) — replaces the fragile iloc[:-HORIZON]
    indicator_cols = ["EMA_20", "RSI", "MACD", "ATR", "OBV", "Direction",
                      "Return_1d_lag1"]
    df.dropna(subset=indicator_cols, inplace=True)

    # ── Date column ───────────────────────────────────────────────────────
    df.reset_index(inplace=True)
    if df.columns[0] != "Date":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Drop raw (leaking) return columns before returning
    df.drop(columns=["Return_1d", "Return_5d"], inplace=True, errors="ignore")

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD TECHNICAL DATASET (FIXED v3)")
    print(f"  Date range         : {START_DATE} -> {END_DATE}")
    print(f"  Prediction horizon : {PREDICTION_HORIZON} day(s)")
    print(f"  Direction type     : {'Ternary {-1,0,1}' if TERNARY_DIRECTION else 'Binary {0,1}'}")
    print(f"  Flat threshold     : {FLAT_THRESHOLD} (5-day return)")
    print(f"  Stocks             : {len(STOCKS)}")
    print(f"  Output             : {OUTPUT_FILE}")
    print("=" * 60)

    all_data        = []
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
        time.sleep(0.8)

    if not all_data:
        print("\n[ERROR] No data downloaded.")
        return

    final = pd.concat(all_data, ignore_index=True)
    final.sort_values(by=["Date", "Stock"], inplace=True)
    final.reset_index(drop=True, inplace=True)

    # Keep only defined output columns that actually exist
    cols  = [c for c in OUTPUT_COLUMNS if c in final.columns]
    final = final[cols]
    final.to_csv(OUTPUT_FILE, index=False)

    pos_rate = (final["Direction"] == 1).mean()
    print("\n" + "=" * 60)
    print(f"  [OK] Saved -> {OUTPUT_FILE}")
    print(f"     Shape      : {final.shape}")
    print(f"     Date range : {final['Date'].min()} -> {final['Date'].max()}")
    print(f"     Stocks     : {final['Stock'].nunique()}  "
          f"(success={success}, skip={skipped})")
    if TERNARY_DIRECTION:
        neu_rate = (final["Direction"] == 0).mean()
        print(f"     Direction  : UP={pos_rate:.1%}  FLAT={neu_rate:.1%}  "
              f"DOWN={1-pos_rate-neu_rate:.1%}")
    else:
        print(f"     Direction  : UP={pos_rate:.1%}  DOWN={1-pos_rate:.1%}")
    print(f"     Horizon    : {PREDICTION_HORIZON}-day forward return")
    print("=" * 60)


if __name__ == "__main__":
    main()