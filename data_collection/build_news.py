# ============================================================
# build_news.py
# Scrapes financial news headlines from Moneycontrol for
# NIFTY 50 stocks within a specified date range.
# Outputs: data/news/news.csv
# Install: pip install requests beautifulsoup4 pandas
# ============================================================

import os, re, time, warnings
import requests
import pandas as pd
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

START_DATE  = "2015-01-01"
END_DATE    = "2025-12-31"
STOCK_COUNT = 40
MAX_ARTICLES_PER_STOCK  = 150
MIN_ARTICLES_THRESHOLD  = 2

FY_START = pd.to_datetime(START_DATE)
FY_END   = pd.to_datetime(END_DATE)

_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "data", "news")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news.csv")

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

MC_SLUG_MAP = {
    "HDFCBANK":   "hdfc-bank",
    "ICICIBANK":  "icici-bank",
    "SBIN":       "state-bank-of-india",
    "AXISBANK":   "axis-bank",
    "KOTAKBANK":  "kotak-mahindra-bank",
    "BAJFINANCE": "bajaj-finance",
    "BAJAJFINSV": "bajaj-finserv",
    "INDUSINDBK": "indusind-bank",
    "TCS":        "tcs",
    "INFY":       "infosys",
    "HCLTECH":    "hcl-technologies",
    "WIPRO":      "wipro",
    "TECHM":      "tech-mahindra",
    "RELIANCE":   "reliance-industries",
    "ONGC":       "ongc",
    "NTPC":       "ntpc",
    "POWERGRID":  "power-grid-corporation-of-india",
    "BPCL":       "bpcl",
    "HINDUNILVR": "hindustan-unilever",
    "ITC":        "itc",
    "NESTLEIND":  "nestle-india",
    "BRITANNIA":  "britannia",
    "MARUTI":     "maruti-suzuki",
    "M&M":        "mahindra-and-mahindra",
    "BHARTIARTL": "bharti-airtel",
    "EICHERMOT":  "eicher-motors",
    "HEROMOTOCO": "hero-motocorp",
    "BAJAJ-AUTO": "bajaj-auto",
    "SUNPHARMA":  "sun-pharmaceutical-industries",
    "CIPLA":      "cipla",
    "DRREDDY":    "dr-reddys-laboratories",
    "TATASTEEL":  "tata-steel",
    "JSWSTEEL":   "jsw-steel",
    "HINDALCO":   "hindalco",
    "COALINDIA":  "coal-india",
    "LT":         "larsen-and-toubro",
    "ULTRACEMCO": "ultratech-cement",
    "GRASIM":     "grasim",
    "ASIANPAINT": "asian-paints",
    "TITAN":      "titan",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_DATE_PATTERN = re.compile(r"<span[^>]*>(.*?)</span>")
_IST_PATTERN  = re.compile(r"IST.*")


def fetch_news(stock: str, session: requests.Session) -> list:
    slug         = MC_SLUG_MAP.get(stock, stock.lower())
    articles     = []
    stale_streak = 0

    print(f"  {stock:<14}", end="", flush=True)

    for page in range(1, 101):
        url = (
            f"https://www.moneycontrol.com/news/tags/{slug}.html"
            if page == 1
            else f"https://www.moneycontrol.com/news/tags/{slug}/page-{page}/"
        )
        try:
            response = session.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 404:
                break
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            article_list = soup.find("ul", id="cagetory")
            if not article_list:
                break
            items = article_list.find_all("li", class_="clearfix")
            if not items:
                break

            for item in items:
                h2 = item.find("h2")
                if not h2:
                    continue
                date_match = _DATE_PATTERN.search(str(item))
                if not date_match:
                    continue
                raw_date = _IST_PATTERN.sub("", date_match.group(1)).strip()
                try:
                    parsed_date = pd.to_datetime(raw_date)
                except (ValueError, TypeError):
                    continue

                if parsed_date < FY_START:
                    stale_streak += 1
                    continue
                if parsed_date > FY_END:
                    continue

                stale_streak = 0
                articles.append({
                    "Date"     : parsed_date.strftime("%Y-%m-%d"),
                    "Stock"    : stock,
                    "News_Text": h2.get_text(strip=True),
                    "Source"   : "Moneycontrol",
                })

            if stale_streak >= 30:
                break

        except requests.exceptions.RequestException:
            break
        except Exception:
            break

        time.sleep(0.5)

    count = len(articles)
    if count < MIN_ARTICLES_THRESHOLD:
        print(f"[SKIP] Only {count} articles (min {MIN_ARTICLES_THRESHOLD})")
        return []

    print(f"[OK] {count} articles")
    return articles[:MAX_ARTICLES_PER_STOCK]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["News_Text", "Date"])
    df["News_Text"] = (
        df["News_Text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    df = df[df["News_Text"].str.len() >= 10]
    df = df.drop_duplicates(subset=["News_Text"], keep="first")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[(df["Date"] >= FY_START) & (df["Date"] <= FY_END)]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df.sort_values(by=["Date", "Stock", "News_Text"]).reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print(f"  BUILD NEWS DATASET")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"  Stocks     : {len(STOCKS)}")
    print(f"  Output     : {OUTPUT_FILE}")
    print("=" * 60)

    all_articles = []
    successful   = []

    try:
        with requests.Session() as session:
            for stock in STOCKS:
                news = fetch_news(stock, session)
                if news:
                    all_articles.extend(news)
                    successful.append(stock)
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    if not all_articles:
        print("\n[ERROR] No articles collected.")
        return

    df_clean = clean_data(pd.DataFrame(all_articles))
    if df_clean.empty:
        print("\n[ERROR] No articles survived cleaning.")
        return

    df_clean[["Date", "Stock", "News_Text", "Source"]].to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ Saved → {OUTPUT_FILE}")
    print(f"     Articles   : {len(df_clean)}")
    print(f"     Date range : {df_clean['Date'].min()} → {df_clean['Date'].max()}")
    print(f"     Stocks     : {', '.join(successful)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
