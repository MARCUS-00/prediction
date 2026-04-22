# ============================================================
# build_events.py
# Constructs a daily event dataset combining macro-economic
# and corporate events for NIFTY 50 stocks using NSE data.
# Outputs: data/events/events.csv
# Install: pip install pandas nsepython
# ============================================================

import os, time
import pandas as pd
from collections import defaultdict

try:
    from nsepython import nsefetch
except ImportError:
    raise ImportError("Please install nsepython: pip install nsepython")

START_DATE  = "2015-01-01"
END_DATE    = "2025-12-31"
STOCK_COUNT = 40

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
    "DIVIDEND":      0.7,
    "BOARD_MEETING": 0.7,
    "NONE":          0.0,
}

_ALL_RBI_POLICY_DATES = {
    "2010-01-29", "2010-03-19", "2010-04-20", "2010-07-27", "2010-09-16", "2010-11-02",
    "2011-01-25", "2011-03-17", "2011-05-03", "2011-06-16", "2011-07-26", "2011-09-16", "2011-10-25", "2011-12-16",
    "2012-01-24", "2012-03-15", "2012-04-17", "2012-06-18", "2012-07-31", "2012-09-17", "2012-10-30", "2012-12-18",
    "2013-01-29", "2013-03-19", "2013-05-03", "2013-06-17", "2013-07-30", "2013-09-20", "2013-10-29", "2013-12-18",
    "2014-01-28", "2014-02-28", "2014-04-01", "2014-06-03", "2014-08-05", "2014-09-30", "2014-12-02",
    "2015-01-15", "2015-02-03", "2015-04-07", "2015-06-02", "2015-08-04", "2015-09-29", "2015-12-01",
    "2016-02-02", "2016-04-05", "2016-06-07", "2016-08-09", "2016-10-04", "2016-12-07",
    "2017-02-08", "2017-04-06", "2017-06-07", "2017-08-02", "2017-10-04", "2017-12-06",
    "2018-02-07", "2018-04-05", "2018-06-06", "2018-08-01", "2018-10-05", "2018-12-05",
    "2019-02-07", "2019-04-04", "2019-06-06", "2019-08-07", "2019-10-04", "2019-12-05",
    "2020-02-06", "2020-03-27", "2020-05-22", "2020-08-06", "2020-10-09", "2020-12-04",
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06", "2021-10-08", "2021-12-08",
    "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-08", "2025-12-05",
}

_ALL_REPO_RATE_CHANGE_DATES = {
    "2010-01-29", "2010-03-19", "2010-04-20", "2010-07-27", "2010-09-16", "2010-11-02",
    "2011-01-25", "2011-03-17", "2011-05-03", "2011-06-16", "2011-07-26",
    "2012-04-17",
    "2013-01-29", "2013-03-19", "2013-05-03",
    "2014-01-28",
    "2015-01-15", "2015-02-03", "2015-04-07", "2015-06-02", "2015-09-29",
    "2016-04-05",
    "2019-02-07", "2019-04-04", "2019-06-06", "2019-08-07", "2019-10-04",
    "2020-03-27", "2020-05-22",
    "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08",
    "2025-02-07", "2025-04-09",
}


def _filter_dates_to_range(date_set: set, start: str, end: str) -> set:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    result = set()
    for d in date_set:
        try:
            if s <= pd.to_datetime(d) <= e:
                result.add(d)
        except Exception:
            pass
    return result


def fetch_corporate_events(symbol: str, start_date: str, end_date: str) -> list:
    events      = []
    current     = pd.to_datetime(start_date)
    final_end   = pd.to_datetime(end_date)
    error_count = 0

    print(f"  {symbol:<14}", end="", flush=True)

    while current <= final_end:
        chunk_end = min(current + pd.DateOffset(months=3), final_end)
        s_str = current.strftime("%d-%m-%Y")
        e_str = chunk_end.strftime("%d-%m-%Y")

        corp_url  = (f"https://www.nseindia.com/api/corporate-announcements"
                     f"?index=equities&symbol={symbol}&from_date={s_str}&to_date={e_str}")
        board_url = (f"https://www.nseindia.com/api/event-calendar"
                     f"?index=equities&symbol={symbol}&from_date={s_str}&to_date={e_str}")

        for url in [corp_url, board_url]:
            try:
                data = nsefetch(url)
                if isinstance(data, list):
                    events.extend(data)
                elif isinstance(data, dict) and "data" in data:
                    events.extend(data["data"])
                print(".", end="", flush=True)
            except Exception:
                error_count += 1
                print("x", end="", flush=True)

        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)

    if len(events) < 5:
        print(f" [WARN: {len(events)} API events -> fallback]", end="")
        events.extend(_build_fallback_events(symbol, start_date, end_date))

    print(f" [DONE: {len(events)} events]")
    return events


def _build_fallback_events(symbol: str, start_date: str, end_date: str) -> list:
    KNOWN_SCHEDULES = {
        "TCS":       {"months": [1, 4, 7, 10], "day": 10},
        "INFY":      {"months": [1, 4, 7, 10], "day": 13},
        "HDFCBANK":  {"months": [1, 4, 7, 10], "day": 16},
        "RELIANCE":  {"months": [1, 4, 7, 10], "day": 21},
        "ICICIBANK": {"months": [1, 4, 7, 10], "day": 23},
        "SBIN":      {"months": [1, 4, 7, 10], "day": 28},
    }
    cfg        = KNOWN_SCHEDULES.get(symbol, {"months": [1, 4, 7, 10], "day": 10 + (len(symbol) % 9)})
    start_year = pd.to_datetime(start_date).year
    end_year   = pd.to_datetime(end_date).year
    fallback   = []

    def _next_weekday(ts):
        if ts.weekday() == 5: return ts + pd.Timedelta(days=2)
        if ts.weekday() == 6: return ts + pd.Timedelta(days=1)
        return ts

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
    if any(k in p for k in ["FINANCIAL RESULTS", "EARNINGS"]): return "EARNINGS"
    if "DIVIDEND"      in p: return "DIVIDEND"
    if "BOARD MEETING" in p: return "BOARD_MEETING"
    if "SPLIT"         in p: return "STOCK_SPLIT"
    if "BONUS"         in p: return "BONUS"
    if "MERGER"        in p: return "MERGER"
    return "NONE"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RBI_POLICY_DATES = _filter_dates_to_range(_ALL_RBI_POLICY_DATES,       START_DATE, END_DATE)
    REPO_RATE_DATES  = _filter_dates_to_range(_ALL_REPO_RATE_CHANGE_DATES,  START_DATE, END_DATE)

    print("=" * 60)
    print(f"  BUILD EVENTS DATASET")
    print(f"  Date range       : {START_DATE} -> {END_DATE}")
    print(f"  Stocks           : {len(STOCKS)}")
    print(f"  RBI policy dates : {len(RBI_POLICY_DATES)} in range")
    print(f"  Repo rate dates  : {len(REPO_RATE_DATES)} in range")
    print(f"  Output           : {OUTPUT_FILE}")
    print("=" * 60)

    trading_days = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    ts           = trading_days.to_series()

    pmi_dates          = set(ts.groupby([ts.dt.year, ts.dt.month]).first().dt.strftime("%Y-%m-%d"))
    cpi_dates          = set(ts[ts.dt.day >= 12].groupby([ts.dt.year, ts.dt.month]).first().dt.strftime("%Y-%m-%d"))
    gdp_dates          = set(ts[ts.dt.month.isin([2, 5, 8, 11])].groupby([ts.dt.year, ts.dt.month]).last().dt.strftime("%Y-%m-%d"))
    union_budget_dates = set(ts[ts.dt.month == 2].groupby(ts.dt.year).first().dt.strftime("%Y-%m-%d"))

    print("\n  Fetching NSE corporate events:")
    print("  " + "-" * 50)
    corporate_event_map = defaultdict(list)

    for symbol in STOCKS:
        raw = fetch_corporate_events(symbol, START_DATE, END_DATE)
        for item in raw:
            date_val = item.get("date") or item.get("bm_date") or item.get("exDate")
            if not date_val:
                continue
            try:
                parsed = pd.to_datetime(date_val).strftime("%Y-%m-%d")
            except Exception:
                continue
            event_type = _classify_event(str(item.get("purpose", item.get("subject", ""))))
            if event_type != "NONE":
                key = (parsed, symbol)
                if event_type not in corporate_event_map[key]:
                    corporate_event_map[key].append(event_type)

    print("\n  Constructing final dataset (macro + corporate) ...")
    rows = []

    for date in trading_days:
        date_str = date.strftime("%Y-%m-%d")
        for symbol in STOCKS:
            day_events = []
            categories = set()

            if date_str in union_budget_dates: day_events.append("UNION_BUDGET"); categories.add("GOVT")
            if date_str in RBI_POLICY_DATES:   day_events.append("RBI_POLICY");   categories.add("GOVT")
            if date_str in REPO_RATE_DATES:    day_events.append("REPO_RATE");    categories.add("GOVT")
            if date_str in gdp_dates:          day_events.append("GDP");          categories.add("MACRO")
            if date_str in cpi_dates:          day_events.append("CPI");          categories.add("MACRO")
            if date_str in pmi_dates:          day_events.append("PMI");          categories.add("MACRO")

            for ev in corporate_event_map.get((date_str, symbol), []):
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

            rows.append({
                "date"           : date_str,
                "symbol"         : symbol,
                "event_category" : event_category,
                "event_name"     : event_name,
                "event_score_max": event_score,
                "event_count"    : len(day_events),
                "is_event"       : 1 if day_events else 0,
            })

    df = pd.DataFrame(rows, columns=[
        "date", "symbol", "event_category", "event_name",
        "event_score_max", "event_count", "is_event",
    ])
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"  [OK] Saved -> {OUTPUT_FILE}")
    print(f"     Rows        : {len(df)}")
    print(f"     Event-days  : {df['is_event'].sum()}")
    print(f"     Date range  : {df['date'].min()} -> {df['date'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
