import os, sys, time
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATE_START, DATE_END

try:
    from nsepython import nsefetch
except ImportError:
    raise ImportError("Please install nsepython: pip install nsepython")

START_DATE  = DATE_START
END_DATE    = DATE_END
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
    "2020-02-06", "2020-03-27", "2020-05-22", "2020-08-06", "2020-10-09", "2020-12-04",
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06", "2021-10-08", "2021-12-08",
    "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-08", "2025-12-05",
}

_ALL_REPO_RATE_CHANGE_DATES = {
    "2020-03-27", "2020-05-22",
    "2022-05-04", "2022-06-08", "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08",
    "2025-02-07", "2025-04-09",
}


def _filter_dates_to_range(date_set, start, end):
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    result = set()
    for d in date_set:
        try:
            if s <= pd.to_datetime(d) <= e:
                result.add(d)
        except Exception:
            pass
    return result


def fetch_corporate_events(symbol, start_date, end_date):
    events    = []
    current   = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)
    print(f"  {symbol:<14}", end="", flush=True)

    while current <= final_end:
        chunk_end = min(current + pd.DateOffset(months=3), final_end)
        s_str = current.strftime("%d-%m-%Y")
        e_str = chunk_end.strftime("%d-%m-%Y")
        corp_url = (f"https://www.nseindia.com/api/corporate-announcements"
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
                print("x", end="", flush=True)
        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)

    print(f" [DONE: {len(events)} API events]")
    return events


def _classify_event(raw_purpose):
    p = str(raw_purpose).upper()
    if any(k in p for k in ["FINANCIAL RESULTS", "EARNINGS"]):
        return "EARNINGS"
    if "DIVIDEND"      in p: return "DIVIDEND"
    if "BOARD MEETING" in p: return "BOARD_MEETING"
    if "SPLIT"         in p: return "STOCK_SPLIT"
    if "BONUS"         in p: return "BONUS"
    if "MERGER"        in p: return "MERGER"
    return "NONE"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RBI_POLICY_DATES = _filter_dates_to_range(_ALL_RBI_POLICY_DATES,      START_DATE, END_DATE)
    REPO_RATE_DATES  = _filter_dates_to_range(_ALL_REPO_RATE_CHANGE_DATES, START_DATE, END_DATE)

    print("=" * 60)
    print("  BUILD EVENTS DATASET")
    print(f"  Date range       : {START_DATE} -> {END_DATE}")
    print(f"  Stocks           : {len(STOCKS)}")
    print(f"  RBI policy dates : {len(RBI_POLICY_DATES)} in range")
    print(f"  Repo rate dates  : {len(REPO_RATE_DATES)} in range")
    print(f"  Output           : {OUTPUT_FILE}")
    print("=" * 60)

    trading_days = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    ts = trading_days.to_series()

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
            event_type = _classify_event(item.get("purpose", item.get("subject", "")))
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
                is_event       = 1
            else:
                event_name = event_category = "NONE"
                event_score = 0.0
                is_event    = 0
            rows.append({
                "date":            date_str,
                "symbol":          symbol,
                "event_category":  event_category,
                "event_name":      event_name,
                "event_score_max": event_score,
                "event_count":     len(day_events),
                "is_event":        is_event,
            })

    df = pd.DataFrame(rows, columns=[
        "date", "symbol", "event_category", "event_name",
        "event_score_max", "event_count", "is_event",
    ])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved -> {OUTPUT_FILE}  rows={len(df)}  event_days={int(df['is_event'].sum())}")


if __name__ == "__main__":
    main()
