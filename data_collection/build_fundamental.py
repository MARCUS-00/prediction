# ============================================================
# build_fundamental.py  (FIXED v5)
#
# Fixes vs v4:
#   1. [CRITICAL] expand_years() bfill on non-growth columns could
#      propagate the EARLIEST available fundamental data into years
#      before the company first reported (e.g. 2020 filled with 2021
#      data for stocks that IPO'd in 2021).  This is look-ahead leakage.
#      Fixed: bfill is now only applied UP TO the first reported year;
#      years before the first available report stay NaN (they will be
#      imputed downstream with the cross-stock median, not a future value
#      from the same stock).
#
#   2. [CRITICAL] report_available_from was set to May 1 of year+1 for
#      every row in the raw data.  After expand_years() called ffill on
#      it, rows created for years before any real data had
#      report_available_from = NaN, which the merge step treated as
#      "always available" rather than "never available".
#      Fixed: rows with no real data (all fundamental cols NaN) get
#      report_available_from = "9999-12-31" (a sentinel that no date
#      will be less than), making merge_features correctly exclude them.
#
#   3. [BUG] expand_years() was called on the whole DataFrame but
#      produced duplicate (Stock, Year) rows when the same year appeared
#      in both the full-range template and the fetched data.  Added
#      explicit deduplication (keep='last', prefer real data) before
#      the merge step inside expand_years.
#
#   4. [BUG] EPS fallback used info["sharesOutstanding"] which is a
#      current snapshot, not the historical share count.  The fallback
#      is now only used when Basic/Diluted EPS is truly absent from
#      yfinance's own financials table (unchanged), but a warning is
#      now printed so users know this path was taken, since the computed
#      EPS may be inaccurate for historical years.
#
#   5. [MINOR] Added a post-build constant-column check.  Any feature
#      column whose non-NaN values have std == 0 across the whole
#      dataset is flagged as a warning (would be useless in a model).
#
#   6. [MINOR] Added Revenue and Profit normalisation columns:
#      Revenue_B  = Revenue  / 1e9  (in billions INR)
#      Profit_B   = Profit   / 1e9
#      These are more numerically stable for tree models than raw INR.
# ============================================================

import os, sys
import numpy as np
import pandas as pd
import yfinance as yf

# ── Date range ────────────────────────────────────────────────────────────────
PRICE_START_DATE = "2020-01-01"
PRICE_END_DATE   = "2025-12-31"
START_YEAR       = 2020
END_YEAR         = 2025
STOCK_COUNT      = 40

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

try:
    from config.settings import SECTOR_MAP
except ImportError:
    SECTOR_MAP = {
        "HDFCBANK":"Financials","ICICIBANK":"Financials","SBIN":"Financials",
        "AXISBANK":"Financials","KOTAKBANK":"Financials","BAJFINANCE":"Financials",
        "BAJAJFINSV":"Financials","INDUSINDBK":"Financials",
        "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
        "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy",
        "NTPC":"Utilities","POWERGRID":"Utilities","COALINDIA":"Utilities",
        "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG",
        "BRITANNIA":"FMCG","TITAN":"Consumer",
        "MARUTI":"Auto","M&M":"Auto","EICHERMOT":"Auto",
        "HEROMOTOCO":"Auto","BAJAJ-AUTO":"Auto",
        "SUNPHARMA":"Pharma","CIPLA":"Pharma","DRREDDY":"Pharma",
        "TATASTEEL":"Metals","JSWSTEEL":"Metals","HINDALCO":"Metals",
        "LT":"Infra","ULTRACEMCO":"Cement","GRASIM":"Cement",
        "ASIANPAINT":"Consumer","BHARTIARTL":"Telecom",
    }

OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "fundamental")
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

_OUTLIER_CLIP = 1000.0

# Sentinel for "no fundamental data available" — merge_features should
# treat any date < 9999-12-31 as before-data, i.e. exclude the row.
_NO_DATA_SENTINEL = "9999-12-31"


def safe_get(df, keys: list, col) -> float:
    if df is None or col is None:
        return np.nan
    for k in keys:
        if k in df.index:
            try:
                val = df.loc[k, col]
                if not pd.isna(val):
                    return float(val)
            except (KeyError, TypeError):
                continue
    return np.nan


def find_col(df, year: int):
    if df is None or df.empty:
        return None
    for c in df.columns:
        try:
            col_year = pd.to_datetime(c).year
            if col_year == year:
                return c
        except Exception:
            continue
    return None


def _clip(val: float) -> float:
    if pd.isna(val):
        return np.nan
    return val if abs(val) <= _OUTLIER_CLIP else np.nan


def get_stock_data(stock: str) -> list:
    base = stock.replace(".NS", "")
    data = []
    try:
        ticker = yf.Ticker(stock)

        hist = ticker.history(start=PRICE_START_DATE, end=PRICE_END_DATE)
        if hist.empty:
            print(f"  ⚠ Skipped {stock} (no price data)")
            return []
        hist.index   = pd.to_datetime(hist.index).tz_localize(None)
        hist["Year"] = hist.index.year
        price_map    = hist.groupby("Year")["Close"].last()

        fin  = ticker.financials
        bs   = ticker.balance_sheet
        info = ticker.info or {}

        if fin is None or fin.empty:
            print(f"  ⚠ Skipped {stock} (no financials)")
            return []

        fin.columns = pd.to_datetime(fin.columns)
        if bs is not None and not bs.empty:
            bs.columns = pd.to_datetime(bs.columns)

        shares_outstanding = info.get("sharesOutstanding", np.nan)

        years = sorted({c.year for c in fin.columns})
        for year in years:
            f_col = find_col(fin, year)
            p_col = find_col(fin, year - 1)
            b_col = find_col(bs,  year) if bs is not None else None

            revenue = safe_get(fin, ["Total Revenue"], f_col)
            profit  = safe_get(fin, ["Net Income"],    f_col)

            eps = safe_get(fin, ["Basic EPS", "Diluted EPS"], f_col)
            if np.isnan(eps):
                if shares_outstanding and shares_outstanding > 0 and not np.isnan(profit):
                    # FIX v5: warn that this EPS is current-snapshot-based (inaccurate)
                    print(f"  [WARN] {base} year={year}: EPS computed from current "
                          f"sharesOutstanding — may be inaccurate for historical years")
                    eps = profit / shares_outstanding
                else:
                    eps = np.nan

            price = float(price_map.get(year, np.nan))
            pe    = _clip(price / eps) if (not np.isnan(eps) and eps != 0) else np.nan

            equity = safe_get(bs, ["Stockholders Equity", "Total Stockholder Equity"], b_col)
            debt   = safe_get(bs, ["Total Debt", "Long Term Debt"],                   b_col)

            roe = _clip(profit / equity) if (not np.isnan(equity) and equity != 0) else np.nan
            dte = _clip(debt   / equity) if (not np.isnan(equity) and equity != 0) else np.nan

            rev_g = prof_g = np.nan
            if p_col is not None:
                prev_rev    = safe_get(fin, ["Total Revenue"], p_col)
                prev_profit = safe_get(fin, ["Net Income"],    p_col)
                if not np.isnan(prev_rev)    and prev_rev    != 0:
                    rev_g  = (revenue - prev_rev)    / abs(prev_rev)
                if not np.isnan(prev_profit) and prev_profit != 0:
                    prof_g = (profit  - prev_profit) / abs(prev_profit)

            report_available = pd.Timestamp(year=year + 1, month=5, day=1).strftime("%Y-%m-%d")

            data.append({
                "Stock"               : base,
                "Year"                : year,
                "Sector"              : SECTOR_MAP.get(base, "Unknown"),
                "PE_Ratio"            : pe,
                "EPS"                 : eps,
                "ROE"                 : roe,
                "Debt_to_Equity"      : dte,
                "Revenue"             : revenue,
                "Profit"              : profit,
                "Revenue_Growth"      : rev_g,
                "Profit_Growth"       : prof_g,
                "report_available_from": report_available,
            })

    except Exception as e:
        print(f"  [ERROR] {stock}: {e}")
    return data


def expand_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each stock to the full START_YEAR..END_YEAR range.

    FIX v5 changes:
      - bfill on non-growth numerics is now REMOVED to prevent look-ahead
        from propagating future data into years before the first report.
        Only ffill is used (fills later NaN years from prior real data).
        Years before the first available report are left as NaN.
      - Duplicate (Stock, Year) rows are deduped before the merge
        (keep='last' preserves real fetched data over the template NaN).
      - Rows with no real fundamental data get report_available_from =
        _NO_DATA_SENTINEL so merge_features excludes them correctly.

    Growth columns  → ffill only (no bfill — same as v4).
    Non-growth cols → ffill only (FIX: was ffill+bfill — bfill removed).
    report_available_from → NaN rows (no data) get sentinel value.
    """
    non_growth_num = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity", "Revenue", "Profit"]
    growth_cols    = ["Revenue_Growth", "Profit_Growth"]
    final          = []

    for stock in df["Stock"].unique():
        sub = df[df["Stock"] == stock].copy()

        # FIX v5: deduplicate (stock, year) before merging with the template
        sub = sub.sort_values("Year").drop_duplicates(subset=["Year"], keep="last")

        full = pd.DataFrame({"Year": range(START_YEAR, END_YEAR + 1)})
        sub  = full.merge(sub, on="Year", how="left")
        sub["Stock"]  = stock
        sub["Sector"] = sub["Sector"].ffill().bfill()
        sub["Sector"] = sub["Sector"].fillna(SECTOR_MAP.get(stock, "Unknown"))
        sub.sort_values("Year", inplace=True)

        # FIX v5: ffill only — do NOT bfill (prevents pre-report look-ahead)
        for col in non_growth_num:
            if sub[col].notna().any():
                sub[col] = sub[col].ffill()
                # bfill intentionally removed here

        # Growth: ffill only — same as v4
        for col in growth_cols:
            sub[col] = sub[col].ffill()

        # report_available_from: ffill for filled years; sentinel for no-data rows
        sub["report_available_from"] = sub["report_available_from"].ffill()
        # FIX v5: rows that still have NaN have no real data → use sentinel
        sub["report_available_from"] = sub["report_available_from"].fillna(_NO_DATA_SENTINEL)

        # FIX v5: normalised revenue/profit (billions INR) — more stable for trees
        sub["Revenue_B"] = sub["Revenue"] / 1e9
        sub["Profit_B"]  = sub["Profit"]  / 1e9

        final.append(sub)

    result = pd.concat(final).sort_values(["Stock", "Year"]).reset_index(drop=True)

    real_pct = result["Revenue_Growth"].notna().mean() * 100
    print(f"  [INFO] Revenue_Growth coverage : {real_pct:.1f}% rows have real values "
          f"(rest are NaN → median-imputed downstream)")
    return result


def _check_constant_columns(df: pd.DataFrame) -> None:
    """
    FIX v5: Warn if any numeric feature column has zero variance
    across all non-NaN values. Such columns provide no signal.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in ("Year",):
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"  [WARN] Column '{col}' is ALL NaN — will be useless in model.")
        elif vals.std() == 0:
            print(f"  [WARN] Column '{col}' is CONSTANT (value={vals.iloc[0]:.4f}) "
                  f"— will be useless in model.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD FUNDAMENTAL DATASET (FIXED v5)")
    print(f"  Date range : {PRICE_START_DATE} -> {PRICE_END_DATE}")
    print(f"  Years      : {START_YEAR} -> {END_YEAR}")
    print(f"  Stocks     : {len(STOCKS)}")
    print("=" * 60)

    dataset          = []
    success = skipped = 0

    for stock in STOCKS:
        print(f"  Processing {stock}...", end=" ", flush=True)
        data = get_stock_data(stock)
        if data:
            dataset.extend(data)
            success += 1
            print("✔")
        else:
            skipped += 1
            print("✘")

    if not dataset:
        print("\n[ERROR] No data collected.")
        return

    df = pd.DataFrame(dataset)
    df = expand_years(df)

    # FIX v5: flag any constant or all-NaN columns before saving
    print("\n  Checking for constant/all-NaN columns...")
    _check_constant_columns(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[OK] Saved -> {OUTPUT_FILE}")
    print(f"  Success : {success}   Skipped : {skipped}   Rows : {len(df)}")
    print(f"  Revenue_Growth non-NaN : {df['Revenue_Growth'].notna().sum()}")
    print(f"  Profit_Growth  non-NaN : {df['Profit_Growth'].notna().sum()}")
    print(f"  Rows with sentinel report_available_from : "
          f"{(df['report_available_from'] == _NO_DATA_SENTINEL).sum()}")
    print(df.tail())


if __name__ == "__main__":
    main()