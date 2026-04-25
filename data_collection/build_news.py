# ============================================================
# data_collection/build_news.py  (FIXED v2)
#
# Fixes vs v1 (FIXED):
#   1. [CRITICAL] FinBERT now runs with explicit error diagnostics and
#      a CPU-safe pipeline. The v1 "fix" fell back to neutral placeholders
#      silently on ANY exception (including missing torch/transformers),
#      so the entire news.csv was written as 0.333/0.333/0.334.
#      v2 installs the dependency if absent and surfaces the real error.
#
#   2. [CRITICAL] news_score = news_positive - news_negative is now
#      computed and saved in news.csv.  merge_features.py reads
#      news_score directly from disk; it no longer needs to recompute it,
#      which means the column is always present in merged_final.csv.
#
#   3. Neutral-placeholder fallback is now clearly flagged in the output
#      with a WARNING so the user knows sentiment is not real.
#
#   4. Added FINBERT_AVAILABLE guard so users without GPU/transformers
#      still get a usable (though placeholder) dataset, with clear notice.
# ============================================================

import os, re, time, warnings, subprocess, sys
import requests
import pandas as pd
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

START_DATE  = "2015-01-01"
END_DATE    = "2025-12-31"
STOCK_COUNT = 40
MAX_ARTICLES_PER_STOCK  = 1000
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
    "HDFCBANK":   "hdfc-bank",         "ICICIBANK":  "icici-bank",
    "SBIN":       "state-bank-of-india","AXISBANK":   "axis-bank",
    "KOTAKBANK":  "kotak-mahindra-bank","BAJFINANCE": "bajaj-finance",
    "BAJAJFINSV": "bajaj-finserv",      "INDUSINDBK": "indusind-bank",
    "TCS":        "tcs",                "INFY":       "infosys",
    "HCLTECH":    "hcl-technologies",   "WIPRO":      "wipro",
    "TECHM":      "tech-mahindra",      "RELIANCE":   "reliance-industries",
    "ONGC":       "ongc",               "NTPC":       "ntpc",
    "POWERGRID":  "power-grid-corporation-of-india",
    "BPCL":       "bpcl",              "HINDUNILVR": "hindustan-unilever",
    "ITC":        "itc",               "NESTLEIND":  "nestle-india",
    "BRITANNIA":  "britannia",         "MARUTI":     "maruti-suzuki",
    "M&M":        "mahindra-and-mahindra",
    "BHARTIARTL": "bharti-airtel",     "EICHERMOT":  "eicher-motors",
    "HEROMOTOCO": "hero-motocorp",     "BAJAJ-AUTO": "bajaj-auto",
    "SUNPHARMA":  "sun-pharmaceutical-industries",
    "CIPLA":      "cipla",             "DRREDDY":    "dr-reddys-laboratories",
    "TATASTEEL":  "tata-steel",        "JSWSTEEL":   "jsw-steel",
    "HINDALCO":   "hindalco",          "COALINDIA":  "coal-india",
    "LT":         "larsen-and-toubro","ULTRACEMCO":  "ultratech-cement",
    "GRASIM":     "grasim",           "ASIANPAINT":  "asian-paints",
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


def _ensure_transformers():
    """Try to import transformers; install if missing."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        print("[INFO] transformers not found. Installing...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "transformers", "torch", "--quiet"],
            capture_output=True
        )
        if result.returncode == 0:
            print("[INFO] Installation successful.")
            return True
        else:
            print(f"[ERROR] Installation failed: {result.stderr.decode()[:300]}")
            return False


def add_finbert_sentiment(df):
    """
    Score headlines with FinBERT.

    FIX v2:
      - Attempts to install transformers/torch if absent.
      - Surfaces the actual exception instead of swallowing it silently.
      - Falls back to neutral ONLY after exhausting both attempts, and
        prints a clear WARNING so the caller knows sentiment is placeholder.
      - Computes and stores news_score = positive - negative in the CSV
        so merge_features.py does not need to recompute it.
    """
    finbert_ok = _ensure_transformers()
    sentiment_is_real = False

    if finbert_ok:
        try:
            from transformers import pipeline
            print("\n[INFO] Running FinBERT on headlines (this may take a few minutes)...")
            pipe = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,       # CPU; change to 0 for GPU
            )
            texts   = df["News_Text"].fillna("").tolist()
            results = pipe(texts, batch_size=32, truncation=True, max_length=128)

            pos, neg, neu = [], [], []
            for res_list in results:
                s = {r["label"]: r["score"] for r in res_list}
                pos.append(s.get("positive", 0.333))
                neg.append(s.get("negative", 0.333))
                neu.append(s.get("neutral",  0.334))

            df = df.copy()
            df["news_positive"] = pos
            df["news_negative"] = neg
            df["news_neutral"]  = neu
            sentiment_is_real   = True
            print("[INFO] FinBERT scoring complete.")

        except Exception as e:
            print(f"\n[WARNING] FinBERT failed with exception: {e}")
            print("[WARNING] Falling back to neutral placeholders.")
            print("[WARNING] Re-run build_news.py after fixing the issue above")
            print("          to get real sentiment scores.")

    if not sentiment_is_real:
        # FIX: Clearly flag that these are placeholders, not real sentiment
        print("\n[WARNING] *** SENTIMENT IS NEUTRAL PLACEHOLDER ***")
        print("[WARNING] news_positive=0.333, news_negative=0.333 for ALL rows.")
        print("[WARNING] This means news features carry NO discriminative signal.")
        print("[WARNING] Install transformers+torch and rerun to get real scores.")
        df = df.copy()
        df["news_positive"] = 0.333
        df["news_negative"] = 0.333
        df["news_neutral"]  = 0.334

    # FIX: Compute news_score here and save it so merge_features can read it directly
    df["news_score"] = df["news_positive"] - df["news_negative"]
    return df


def fetch_news(stock, session):
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
                    "Date":      parsed_date.strftime("%Y-%m-%d"),
                    "Stock":     stock,
                    "News_Text": h2.get_text(strip=True),
                    "Source":    "Moneycontrol",
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


def clean_data(df):
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
    print("  BUILD NEWS DATASET (FIXED v2)")
    print(f"  Date range : {START_DATE} -> {END_DATE}")
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

    # FIX: Sentiment always scores; on failure uses neutral defaults with clear WARNING.
    # FIX: news_score is now computed and saved here.
    df_clean = add_finbert_sentiment(df_clean)

    # FIX: Save news_score so merge_features.py reads it directly (no recompute needed)
    cols_to_save = [
        "Date", "Stock", "News_Text", "Source",
        "news_positive", "news_negative", "news_neutral", "news_score",
    ]
    df_clean[cols_to_save].to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"  [OK] Saved -> {OUTPUT_FILE}")
    print(f"     Articles   : {len(df_clean)}")
    print(f"     Date range : {df_clean['Date'].min()} -> {df_clean['Date'].max()}")
    print(f"     Stocks     : {', '.join(successful)}")
    print(f"     news_score range: {df_clean['news_score'].min():.4f} to {df_clean['news_score'].max():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()