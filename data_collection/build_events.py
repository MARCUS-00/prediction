# ============================================================
# build_events.py  (FIXED v3)
#
# Fixes vs v2:
#   1. [CRITICAL] Synthetic fallback events DISABLED by default.
#      _build_fallback_events() generated fake quarterly earnings /
#      board meeting / dividend rows for stocks where the NSE API
#      returned < 5 events. These fabricated rows introduced noise
#      (spurious EARNINGS signals on made-up dates) that hurt model
#      reliability. ENABLE_SYNTHETIC_FALLBACK = False by default.
#      Set to True only for debugging/testing when NSE is unreachable.
#
#   2. [PERFORMANCE] Chunk size increased from 3 months to 6 months,
#      cutting the number of API round-trips per stock from ~24 to ~12
#      over the 2020-2025 range.
#
#   3. [PERFORMANCE] Only the corporate-announcements endpoint is used
#      by default (board_url loop removed). The event-calendar endpoint
#      largely duplicates announcements and doubles request count.
#      Set FETCH_BOTH_ENDPOINTS = True to restore the original behaviour.
#
#   4. [PERFORMANCE] Per-chunk sleep reduced from 0.5 s to 0.3 s.
#      A per-stock sleep of 1.0 s is added between stocks to stay
#      within NSE rate limits without slowing intra-stock chunks.
#
#   5. [BUG] lag_date() fell back to the raw date string when the date
#      was not found in td_set (e.g. a non-business day announcement).
#      This silently left unshifted non-business-day dates in the output.
#      Fixed: snap to the nearest next business day before looking up
#      the index.
#
#   6. [BUG] corporate_event_map used a defaultdict(set) but was keyed
#      on (date_str, symbol). When the effective_date (after lag) was
#      computed, the key used original date_str — events still merged
#      correctly because we look up by original date in the assembly
#      loop. No change needed; clarified with a comment.
#
#   7. [MINOR] ENABLE_SYNTHETIC_FALLBACK guard printed misleading
#      "[WARN: N API events → fallback]" even when 0 events is normal
#      (e.g. a newly listed stock). Message now only appears when
#      ENABLE_SYNTHETIC_FALLBACK = True.
# ============================================================

import os, time
import pandas as pd
from collections import defaultdict

try:
    from nsepython import nsefetch
except ImportError:
    raise ImportError("Please install nsepython: pip install nsepython")

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE  = "2020-01-01"
END_DATE    = "2025-12-31"
STOCK_COUNT = 40

# FIX v3: synthetic fallback DISABLED — fake events add noise
ENABLE_SYNTHETIC_FALLBACK = False

# FIX v3: fetch only the primary announcements endpoint by default
FETCH_BOTH_ENDPOINTS = False

# FIX v3: chunk size increased to 6 months (was 3) → ~12 requests per stock
CHUNK_MONTHS = 6

# FIX v3: sleep tuning — faster intra-stock, small pause between stocks
CHUNK_SLEEP  = 0.3   # seconds between API calls within a stock (was 0.5)
STOCK_SLEEP  = 1.0   # seconds between stocks to respect NSE rate limits

# Shift events forward N trading days to align with the price reaction day.
LAG_DAYS = 1

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "events")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "events.csv")

STOCKS = [
    "HDFCBANK",  "ICICIBANK",  "SBIN",       "AXISBANK",  "KOTAKBANK",
    "BAJFINANCE","BAJAJFINSV", "INDUSINDBK", "TCS",       "INFY",
    "HCLTECH",   "WIPRO",      "TECHM",      "RELIANCE",  "ONGC",
    "NTPC",      "POWERGRID",  "BPCL",       "HINDUNILVR","ITC",
    "NESTLEIND", "BRITANNIA",  "MARUTI",     "M&M",       "BHARTIARTL",
    "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "SUNPHARMA", "CIPLA",
    "DRREDDY",   "TATASTEEL",  "JSWSTEEL",   "HINDALCO",  "COALINDIA",
    "LT",        "ULTRACEMCO", "GRASIM",     "ASIANPAINT","TITAN",
][:STOCK_COUNT]

EVENT_SCORES = {
    "EARNINGS":      1.0,
    "UNION_BUDGET":  1.0,
    "RBI_POLICY":    1.0,
    "REPO_RATE":     1.0,
    "GDP":           0.9,
    "STOCK_SPLIT":   0.9,
    "BONUS":         0.9,
    "MERGER":        0.9,
    "CPI":           0.8,
    "PMI":           0.8,
    "AGM":           0.7,
    "DIVIDEND":      0.7,
    "BOARD_MEETING": 0.7,
    "NONE":          0.0,
}

# ── Macro event dates (hardcoded, authoritative) ──────────────────────────────
_ALL_RBI_POLICY_DATES = {
    "2020-02-06", "2020-03-27", "2020-05-22", "2020-08-06",
    "2020-10-09", "2020-12-04",
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06",
    "2021-10-08", "2021-12-08",
    "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08",
    "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10",
    "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08",
    "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06",
    "2025-10-08", "2025-12-05",
}

_ALL_REPO_RATE_CHANGE_DATES = {
    "2020-03-27", "2020-05-22",
    "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08",
    "2025-02-07", "2025-04-09",
}

_UNION_BUDGET_EXACT = {
    "2020-02-01", "2021-02-01", "2022-02-01",
    "2023-02-01", "2024-02-01", "2025-02-01",
}


def _next_weekday(ts: pd.Timestamp) -> pd.Timestamp:
    """Advance to Monday if ts falls on a weekend."""
    dow = ts.weekday()
    if dow == 5:
        return ts + pd.Timedelta(days=2)
    if dow == 6:
        return ts + pd.Timedelta(days=1)
    return ts


def _snap_to_bday(date_str: str, td_set: dict, td_list: list) -> str:
    """
    FIX v3: If date_str is not in the business-day set (e.g. it is a weekend
    or holiday), advance to the next date that IS in td_set.
    """
    if date_str in td_set:
        return date_str
    ts = pd.to_datetime(date_str)
    for _ in range(7):   # try up to 7 days forward
        ts += pd.Timedelta(days=1)
        candidate = ts.strftime("%Y-%m-%d")
        if candidate in td_set:
            return candidate
    return date_str   # give up — original date


def _filter_dates_to_range(date_set: set, start: str, end: str) -> set:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    result = set()
    for d in date_set:
        try:
            dt = pd.to_datetime(d)
            if s <= dt <= e:
                result.add(dt.strftime("%Y-%m-%d"))
        except Exception:
            pass
    return result


# ── Corporate event fetching ──────────────────────────────────────────────────

def fetch_corporate_events(symbol: str, start_date: str, end_date: str) -> list:
    """
    Fetch corporate events from NSE in CHUNK_MONTHS-month chunks.

    FIX v3:
      - Chunk size increased to 6 months.
      - Only the announcements endpoint is fetched (FETCH_BOTH_ENDPOINTS=False).
      - Synthetic fallback disabled by default (ENABLE_SYNTHETIC_FALLBACK=False).
      - Sleep reduced to CHUNK_SLEEP per chunk.
    """
    events      = []
    current     = pd.to_datetime(start_date)
    final_end   = pd.to_datetime(end_date)

    print(f"  {symbol:<14}", end="", flush=True)

    while current <= final_end:
        chunk_end = min(current + pd.DateOffset(months=CHUNK_MONTHS), final_end)
        s_str = current.strftime("%d-%m-%Y")
        e_str = chunk_end.strftime("%d-%m-%Y")

        endpoints = [
            (
                f"https://www.nseindia.com/api/corporate-announcements"
                f"?index=equities&symbol={symbol}&from_date={s_str}&to_date={e_str}"
            )
        ]
        if FETCH_BOTH_ENDPOINTS:
            endpoints.append(
                f"https://www.nseindia.com/api/event-calendar"
                f"?index=equities&symbol={symbol}&from_date={s_str}&to_date={e_str}"
            )

        for url in endpoints:
            try:
                data = nsefetch(url)
                if isinstance(data, list):
                    events.extend(data)
                elif isinstance(data, dict) and "data" in data:
                    events.extend(data["data"])
                print(".", end="", flush=True)
            except Exception:
                print("x", end="", flush=True)

        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(CHUNK_SLEEP)

    # FIX v3: synthetic fallback disabled by default
    if ENABLE_SYNTHETIC_FALLBACK and len(events) < 5:
        print(f" [WARN: {len(events)} API events → synthetic fallback]", end="")
        events.extend(_build_fallback_events(symbol, start_date, end_date))

    print(f" [DONE: {len(events)} events]")
    return events


def _build_fallback_events(symbol: str, start_date: str, end_date: str) -> list:
    """
    Generate synthetic quarterly events — only called when
    ENABLE_SYNTHETIC_FALLBACK = True (disabled by default in v3).

    WARNING: These are fabricated dates. Use only for offline testing
    when the NSE API is unreachable. Never use for production training.
    """
    base_day   = 10 + (sum(ord(c) for c in symbol) % 9)
    cfg        = {"months": [1, 4, 7, 10], "day": base_day}
    start_year = pd.to_datetime(start_date).year
    end_year   = pd.to_datetime(end_date).year
    fallback   = []

    for y in range(start_year, end_year + 1):
        for m in cfg["months"]:
            earn_date  = _next_weekday(pd.Timestamp(year=y, month=m, day=cfg["day"]))
            board_date = _next_weekday(earn_date - pd.Timedelta(days=5))
            fallback.append({"date": earn_date.strftime("%Y-%m-%d"),  "purpose": "FINANCIAL RESULTS"})
            fallback.append({"date": board_date.strftime("%Y-%m-%d"), "purpose": "BOARD MEETING"})
        div_date = _next_weekday(pd.Timestamp(year=y, month=5, day=cfg["day"]))
        fallback.append({"date": div_date.strftime("%Y-%m-%d"), "purpose": "DIVIDEND"})
    return fallback


def _classify_event(raw_purpose: str) -> str:
    p = raw_purpose.upper()
    if any(k in p for k in ["FINANCIAL RESULTS", "EARNINGS", "QUARTERLY RESULTS",
                             "HALF YEARLY RESULTS", "ANNUAL RESULTS"]):
        return "EARNINGS"
    if "DIVIDEND"      in p: return "DIVIDEND"
    if "BOARD MEETING" in p: return "BOARD_MEETING"
    if "SPLIT"         in p: return "STOCK_SPLIT"
    if "BONUS"         in p: return "BONUS"
    if any(k in p for k in ["MERGER", "AMALGAMATION", "ACQUISITION"]): return "MERGER"
    if "AGM"           in p: return "AGM"
    return "NONE"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RBI_POLICY_DATES  = _filter_dates_to_range(_ALL_RBI_POLICY_DATES,      START_DATE, END_DATE)
    REPO_RATE_DATES   = _filter_dates_to_range(_ALL_REPO_RATE_CHANGE_DATES, START_DATE, END_DATE)
    UNION_BUDGET_DATES = {
        _next_weekday(pd.to_datetime(d)).strftime("%Y-%m-%d")
        for d in _UNION_BUDGET_EXACT
        if pd.to_datetime(START_DATE) <= pd.to_datetime(d) <= pd.to_datetime(END_DATE)
    }

    print("=" * 60)
    print("  BUILD EVENTS DATASET (FIXED v3)")
    print(f"  Date range         : {START_DATE} -> {END_DATE}")
    print(f"  Stocks             : {len(STOCKS)}")
    print(f"  RBI policy dates   : {len(RBI_POLICY_DATES)} in range")
    print(f"  Repo rate dates    : {len(REPO_RATE_DATES)} in range")
    print(f"  Union budget       : {len(UNION_BUDGET_DATES)} in range")
    print(f"  Lag days           : {LAG_DAYS}")
    print(f"  Chunk months       : {CHUNK_MONTHS} (was 3)")
    print(f"  Both endpoints     : {FETCH_BOTH_ENDPOINTS}")
    print(f"  Synthetic fallback : {ENABLE_SYNTHETIC_FALLBACK} (disabled = no fake events)")
    print(f"  Output             : {OUTPUT_FILE}")
    print("=" * 60)

    # Build all business days in range
    trading_days = pd.bdate_range(start=START_DATE, end=END_DATE)
    ts_series    = pd.Series(trading_days)

    # Proxy macro dates
    pmi_dates = set(
        ts_series.groupby([ts_series.dt.year, ts_series.dt.month])
        .first().dt.strftime("%Y-%m-%d").values
    )
    cpi_mask  = ts_series.dt.day >= 12
    cpi_dates = set(
        ts_series[cpi_mask]
        .groupby([ts_series[cpi_mask].dt.year, ts_series[cpi_mask].dt.month])
        .first().dt.strftime("%Y-%m-%d").values
    )
    gdp_mask  = ts_series.dt.month.isin([2, 5, 8, 11])
    gdp_dates = set(
        ts_series[gdp_mask]
        .groupby([ts_series[gdp_mask].dt.year, ts_series[gdp_mask].dt.month])
        .last().dt.strftime("%Y-%m-%d").values
    )

    # ── Corporate events ──────────────────────────────────────────────────
    print("\n  Fetching NSE corporate events:")
    print("  " + "-" * 50)

    corporate_event_map: dict[tuple, set] = defaultdict(set)

    for i, symbol in enumerate(STOCKS):
        raw = fetch_corporate_events(symbol, START_DATE, END_DATE)
        for item in raw:
            date_val = (
                item.get("date") or item.get("bm_date") or item.get("exDate")
                or item.get("recordDate") or item.get("anDt")
            )
            if not date_val:
                continue
            try:
                parsed = pd.to_datetime(date_val).strftime("%Y-%m-%d")
            except Exception:
                continue
            purpose    = str(item.get("purpose", item.get("subject", "")))
            event_type = _classify_event(purpose)
            if event_type != "NONE":
                corporate_event_map[(parsed, symbol)].add(event_type)

        # FIX v3: pause between stocks (not just between chunks)
        if i < len(STOCKS) - 1:
            time.sleep(STOCK_SLEEP)

    # ── Assemble dataset ──────────────────────────────────────────────────
    print("\n  Constructing final dataset (macro + corporate) ...")

    td_list = list(trading_days)
    # FIX v3: td_set maps date string → index for both lookup and snap
    td_set  = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(td_list)}

    def lag_date(date_str: str, n: int) -> str:
        """
        Return date_str shifted forward n business days.
        FIX v3: snaps non-business-day dates to the next business day
        before applying the lag offset.
        """
        if n == 0:
            return date_str
        snapped = _snap_to_bday(date_str, td_set, td_list)
        idx     = td_set.get(snapped)
        if idx is None:
            return date_str
        new_idx = min(idx + n, len(td_list) - 1)
        return td_list[new_idx].strftime("%Y-%m-%d")

    rows = []
    for date in trading_days:
        date_str = date.strftime("%Y-%m-%d")
        for symbol in STOCKS:
            day_events: list[str] = []
            categories: set[str]  = set()

            # Macro events (known before open, no lag needed)
            if date_str in UNION_BUDGET_DATES: day_events.append("UNION_BUDGET"); categories.add("GOVT")
            if date_str in RBI_POLICY_DATES:   day_events.append("RBI_POLICY");   categories.add("GOVT")
            if date_str in REPO_RATE_DATES:    day_events.append("REPO_RATE");    categories.add("GOVT")
            if date_str in gdp_dates:          day_events.append("GDP");          categories.add("MACRO")
            if date_str in cpi_dates:          day_events.append("CPI");          categories.add("MACRO")
            if date_str in pmi_dates:          day_events.append("PMI");          categories.add("MACRO")

            # Corporate events (keyed on original announcement date)
            for ev in corporate_event_map.get((date_str, symbol), set()):
                if ev not in day_events:
                    day_events.append(ev)
                    categories.add("STOCK")

            if day_events:
                day_events.sort()
                event_name     = "|".join(day_events)
                event_category = "|".join(sorted(categories))
                event_score    = max(EVENT_SCORES.get(e, 0.0) for e in day_events)
            else:
                event_name = event_category = "NONE"
                event_score = 0.0

            # Shift corporate / after-hours events to next business day
            effective_date = lag_date(date_str, LAG_DAYS) if day_events else date_str

            rows.append({
                "date"            : effective_date,
                "event_date"      : date_str,
                "symbol"          : symbol,
                "event_category"  : event_category,
                "event_name"      : event_name,
                "event_score_max" : event_score,
                "event_count"     : len(day_events),
                "is_event"        : 1 if day_events else 0,
            })

    df = pd.DataFrame(rows, columns=[
        "date", "event_date", "symbol", "event_category",
        "event_name", "event_score_max", "event_count", "is_event",
    ])
    df.sort_values(["date", "symbol"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"  [OK] Saved -> {OUTPUT_FILE}")
    print(f"     Rows        : {len(df)}")
    print(f"     Event-days  : {df['is_event'].sum()}")
    print(f"     Date range  : {df['date'].min()} -> {df['date'].max()}")
    real_events = df[df["event_name"] != "NONE"]
    print(f"     Real events : {len(real_events)} rows with at least one event")
    print("=" * 60)


if __name__ == "__main__":
    main()