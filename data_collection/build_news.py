# ============================================================
# build_news.py  (FIXED v4)
#
# Fixes vs v3:
#   1. [CRITICAL] FinBERT neutral-fallback is now FATAL by default.
#      In v3, if FinBERT failed the script silently wrote
#      news_positive=0.333 for every row, making news_score=0 for
#      all articles — a constant feature that carries zero signal.
#      v4 raises a RuntimeError instead of continuing with junk data.
#      Set ALLOW_NEUTRAL_FALLBACK = True to restore legacy behaviour
#      (e.g. for pipeline testing when GPU/model is unavailable).
#
#   2. [CRITICAL] Post-scoring variance check added.
#      After FinBERT runs, news_score std is computed. If it is below
#      NEWS_SCORE_MIN_STD (default 0.05) a loud warning is printed and
#      the script aborts (or warns if ALLOW_NEUTRAL_FALLBACK=True).
#      Catches cases where the model loads but returns near-uniform
#      scores (e.g. wrong model, truncation artefact, GPU OOM fallback).
#
#   3. [BUG] _apply_lag() shifted all dates including ones that had
#      already been shifted in a previous run (if the CSV was re-read).
#      This is an idempotency concern; document that _apply_lag should
#      only be called once on fresh data (added assertion).
#
#   4. [BUG] fetch_news_parallel() used a bare threading.Lock() alias
#      via __import__ — fragile and unclear. Replaced with an explicit
#      import threading at the top.
#
#   5. [BUG] _parse_date_from_item() returned None when the date span
#      contained a timezone string but pd.to_datetime failed due to the
#      format (e.g. "April 27, 2026 IST"). _IST_RE strip was applied
#      to the full text but not to the matched date portion only.
#      Fixed: strip IST suffix from the entire text before parsing.
#
#   6. [MINOR] Added per-stock news_score summary in final report so
#      users can spot stocks with constant or low-variance scores.
#
#   7. [MINOR] Model name printed clearly; if model download fails
#      (no internet) the error message now suggests using a local path.
# ============================================================

import os, re, time, warnings, subprocess, sys, threading
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE  = "2020-01-01"
END_DATE    = "2025-12-31"
STOCK_COUNT = 40

MAX_ARTICLES_PER_STOCK = 1000
MIN_ARTICLES_THRESHOLD = 2
MAX_PAGES              = 60
MAX_RETRIES            = 3
FETCH_WORKERS          = 5
PAGE_SLEEP             = 0.2
FINBERT_BATCH_SIZE     = 64

# FIX v4: abort if FinBERT fails instead of silently writing neutral values
ALLOW_NEUTRAL_FALLBACK = False   # set True only for offline/debug runs

# FIX v4: minimum acceptable standard deviation of news_score
# If FinBERT returns near-constant scores the feature is useless
NEWS_SCORE_MIN_STD = 0.05

# Shift news dates forward N business days to avoid look-ahead
NEWS_LAG_DAYS = 1

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
    "HDFCBANK":   "hdfc-bank",              "ICICIBANK":  "icici-bank",
    "SBIN":       "state-bank-of-india",    "AXISBANK":   "axis-bank",
    "KOTAKBANK":  "kotak-mahindra-bank",    "BAJFINANCE": "bajaj-finance",
    "BAJAJFINSV": "bajaj-finserv",          "INDUSINDBK": "indusind-bank",
    "TCS":        "tcs",                    "INFY":       "infosys",
    "HCLTECH":    "hcl-technologies",       "WIPRO":      "wipro",
    "TECHM":      "tech-mahindra",          "RELIANCE":   "reliance-industries",
    "ONGC":       "ongc",                   "NTPC":       "ntpc",
    "POWERGRID":  "power-grid-corporation-of-india",
    "BPCL":       "bpcl",                   "HINDUNILVR": "hindustan-unilever",
    "ITC":        "itc",                    "NESTLEIND":  "nestle-india",
    "BRITANNIA":  "britannia",              "MARUTI":     "maruti-suzuki",
    "M&M":        "mahindra-and-mahindra",  "BHARTIARTL": "bharti-airtel",
    "EICHERMOT":  "eicher-motors",          "HEROMOTOCO": "hero-motocorp",
    "BAJAJ-AUTO": "bajaj-auto",
    "SUNPHARMA":  "sun-pharmaceutical-industries",
    "CIPLA":      "cipla",                  "DRREDDY":    "dr-reddys-laboratories",
    "TATASTEEL":  "tata-steel",             "JSWSTEEL":   "jsw-steel",
    "HINDALCO":   "hindalco",               "COALINDIA":  "coal-india",
    "LT":         "larsen-and-toubro",      "ULTRACEMCO": "ultratech-cement",
    "GRASIM":     "grasim",                 "ASIANPAINT": "asian-paints",
    "TITAN":      "titan",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_MONTH_NAMES = (
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    "January|February|March|April|June|July|August|September|October|November|December"
)
_DATE_RE = re.compile(
    rf"({_MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}",
    re.IGNORECASE,
)
_IST_RE  = re.compile(r"\s*IST.*$", re.IGNORECASE)


# ── Dependency helpers ────────────────────────────────────────────────────────

def _ensure_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        print("[INFO] transformers not found — installing...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "transformers", "torch", "--quiet"],
            capture_output=True,
        )
        if result.returncode == 0:
            print("[INFO] Installation successful.")
            return True
        print(f"[ERROR] Installation failed:\n{result.stderr.decode()[:400]}")
        return False


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def add_finbert_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score headlines with FinBERT (ProsusAI/finbert).

    FIX v4:
      - If FinBERT fails and ALLOW_NEUTRAL_FALLBACK=False (default),
        raises RuntimeError to abort the pipeline rather than writing
        constant-zero news_score values that have zero predictive signal.
      - After scoring, checks news_score variance. If std < NEWS_SCORE_MIN_STD,
        raises RuntimeError (or warns if ALLOW_NEUTRAL_FALLBACK=True).
    """
    finbert_ok        = _ensure_transformers()
    sentiment_is_real = False
    pos = neg = neu   = None
    finbert_error     = None

    if finbert_ok:
        try:
            from transformers import pipeline as hf_pipeline
            print("\n[INFO] Loading FinBERT model (ProsusAI/finbert)...")
            print("[INFO] If model download fails, set model= to a local path.")
            pipe = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                top_k=None,
                device=-1,          # CPU; set to 0 for CUDA GPU
                batch_size=FINBERT_BATCH_SIZE,
            )

            texts = df["News_Text"].fillna("").tolist()
            n     = len(texts)
            print(f"[INFO] Scoring {n} headlines in batches of {FINBERT_BATCH_SIZE}...")

            results = pipe(texts, truncation=True, max_length=128)

            pos, neg, neu = [], [], []
            for res_list in results:
                s = {r["label"]: r["score"] for r in res_list}
                pos.append(s.get("positive", 0.333))
                neg.append(s.get("negative", 0.333))
                neu.append(s.get("neutral",  0.334))

            sentiment_is_real = True
            print("[INFO] FinBERT scoring complete.")

        except Exception as exc:
            finbert_error = exc
            print(f"\n[WARNING] FinBERT failed: {exc}")

    df = df.copy()

    if sentiment_is_real and pos is not None:
        df["news_positive"] = pos
        df["news_negative"] = neg
        df["news_neutral"]  = neu
        df["news_score"]    = df["news_positive"] - df["news_negative"]

        # FIX v4: variance check — abort if scores are near-constant
        score_std = df["news_score"].std()
        print(f"[INFO] news_score: mean={df['news_score'].mean():.4f}  "
              f"std={score_std:.4f}  "
              f"min={df['news_score'].min():.4f}  "
              f"max={df['news_score'].max():.4f}")

        if score_std < NEWS_SCORE_MIN_STD:
            msg = (
                f"news_score std={score_std:.4f} is below threshold "
                f"{NEWS_SCORE_MIN_STD}. FinBERT may have loaded incorrectly "
                f"or all headlines are being classified as neutral. "
                f"Check the model and input data."
            )
            if ALLOW_NEUTRAL_FALLBACK:
                print(f"\n[WARNING] {msg}")
                print("[WARNING] Continuing because ALLOW_NEUTRAL_FALLBACK=True.")
            else:
                raise RuntimeError(f"[FATAL] {msg}\n"
                                   f"Set ALLOW_NEUTRAL_FALLBACK=True to override.")

    else:
        # FIX v4: fatal by default — constant news_score kills the feature
        msg = (
            "FinBERT did not produce real sentiment scores. "
            f"Error: {finbert_error}. "
            "Writing neutral placeholders would make news_score=0 for all rows "
            "(zero signal). Fix FinBERT or set ALLOW_NEUTRAL_FALLBACK=True."
        )
        if ALLOW_NEUTRAL_FALLBACK:
            print(f"\n[WARNING] {msg}")
            print("[WARNING] *** SENTIMENT IS NEUTRAL PLACEHOLDER — NO SIGNAL ***")
            df["news_positive"] = 0.333
            df["news_negative"] = 0.333
            df["news_neutral"]  = 0.334
            df["news_score"]    = 0.0   # explicit zero, not positive-negative
        else:
            raise RuntimeError(f"[FATAL] {msg}")

    return df


# ── Date lag helper ───────────────────────────────────────────────────────────

def _next_bday(date_str: str) -> str:
    ts  = pd.to_datetime(date_str)
    ts += pd.tseries.offsets.BDay(1)
    return ts.strftime("%Y-%m-%d")


def _apply_lag(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Shift the Date column forward by `lag` business days.
    FIX v4: asserts 'news_date' column does not already exist to
    prevent double-lagging if this function is called twice.
    """
    if lag <= 0:
        return df
    if "news_date" in df.columns:
        raise ValueError(
            "_apply_lag() called on a DataFrame that already has a "
            "'news_date' column. This means lag was already applied. "
            "Check your pipeline for duplicate calls."
        )
    df = df.copy()
    df["news_date"]  = df["Date"]
    df["Date"]       = df["Date"].apply(
        lambda d: (pd.to_datetime(d) + pd.tseries.offsets.BDay(lag)).strftime("%Y-%m-%d")
    )
    return df


# ── Scraping ──────────────────────────────────────────────────────────────────

def _parse_date_from_item(item) -> "pd.Timestamp | None":
    """
    Extract and parse a date from a BeautifulSoup article <li> element.
    FIX v4: strip IST suffix before attempting to parse, not after.
    """
    for span in item.find_all("span"):
        text = span.get_text(strip=True)
        # FIX v4: strip IST/timezone suffix early so the regex and parser
        # see a clean date string (e.g. "April 27, 2026" not "April 27, 2026 IST")
        text_clean = _IST_RE.sub("", text).strip()
        m = _DATE_RE.search(text_clean)
        if m:
            try:
                return pd.to_datetime(text_clean, dayfirst=False)
            except (ValueError, TypeError):
                continue
    return None


def _build_page_urls(slug: str, page: int) -> list:
    if page == 1:
        return [f"https://www.moneycontrol.com/news/tags/{slug}.html"]
    return [
        f"https://www.moneycontrol.com/news/tags/{slug}/page-{page}/",
        f"https://www.moneycontrol.com/news/tags/{slug}/page/{page}/",
    ]


def fetch_news(stock: str, session: requests.Session) -> list:
    slug         = MC_SLUG_MAP.get(stock, stock.lower().replace("&", "and"))
    articles: list = []
    stale_streak = 0

    for page in range(1, MAX_PAGES + 1):
        urls    = _build_page_urls(slug, page)
        fetched = False

        for url in urls:
            for attempt in range(MAX_RETRIES):
                try:
                    resp = session.get(url, headers=HEADERS, timeout=12)
                    if resp.status_code == 404:
                        return articles[:MAX_ARTICLES_PER_STOCK]
                    resp.raise_for_status()
                    fetched = True
                    break
                except requests.exceptions.RequestException:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(0.3 * (2 ** attempt))
                    continue
            if fetched:
                break

        if not fetched:
            break

        soup         = BeautifulSoup(resp.content, "html.parser")
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
            parsed_date = _parse_date_from_item(item)
            if parsed_date is None:
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

        if stale_streak >= 30 or len(articles) >= MAX_ARTICLES_PER_STOCK:
            break

        time.sleep(PAGE_SLEEP)

    return articles[:MAX_ARTICLES_PER_STOCK]


def fetch_news_parallel(stocks: list) -> tuple:
    """
    Fetch news for all stocks in parallel.
    FIX v4: uses explicit threading.Lock() instead of __import__ alias.
    """
    all_articles: list = []
    successful:   list = []
    lock_print = threading.Lock()   # FIX v4: explicit import

    def _worker(stock: str) -> tuple:
        with requests.Session() as sess:
            arts = fetch_news(stock, sess)
        return stock, arts

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_worker, s): s for s in stocks}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                _, arts = future.result()
            except Exception as exc:
                with lock_print:
                    print(f"  {stock:<14} [ERROR] {exc}")
                continue

            with lock_print:
                if len(arts) < MIN_ARTICLES_THRESHOLD:
                    print(f"  {stock:<14} [SKIP]  {len(arts)} articles "
                          f"(min {MIN_ARTICLES_THRESHOLD})")
                else:
                    print(f"  {stock:<14} [OK]    {len(arts)} articles")
                    all_articles.extend(arts)
                    successful.append(stock)

    return all_articles, successful


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["News_Text", "Date"]).copy()
    df["News_Text"] = (
        df["News_Text"].astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df = df[df["News_Text"].str.len() >= 10]
    df = df.drop_duplicates(subset=["News_Text"], keep="first")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[(df["Date"] >= FY_START) & (df["Date"] <= FY_END)]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(by=["Date", "Stock", "News_Text"]).reset_index(drop=True)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BUILD NEWS DATASET (FIXED v4)")
    print(f"  Date range         : {START_DATE} -> {END_DATE}")
    print(f"  Stocks             : {len(STOCKS)}")
    print(f"  Workers            : {FETCH_WORKERS}  (parallel scraping)")
    print(f"  Max pages/stock    : {MAX_PAGES}")
    print(f"  Lag days           : {NEWS_LAG_DAYS}")
    print(f"  Allow neutral fall : {ALLOW_NEUTRAL_FALLBACK}")
    print(f"  Min score std      : {NEWS_SCORE_MIN_STD}")
    print(f"  Output             : {OUTPUT_FILE}")
    print("=" * 60)

    try:
        all_articles, successful = fetch_news_parallel(STOCKS)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return

    if not all_articles:
        print("\n[ERROR] No articles collected.")
        return

    df_clean = clean_data(pd.DataFrame(all_articles))
    if df_clean.empty:
        print("\n[ERROR] No articles survived cleaning.")
        return

    # Apply business-day lag before sentiment so the date column is final
    df_clean = _apply_lag(df_clean, NEWS_LAG_DAYS)

    # FIX v4: FinBERT scoring — will raise on failure unless ALLOW_NEUTRAL_FALLBACK=True
    df_clean = add_finbert_sentiment(df_clean)

    cols_to_save = [
        "Date", "news_date", "Stock", "News_Text", "Source",
        "news_positive", "news_negative", "news_neutral", "news_score",
    ]
    cols_to_save = [c for c in cols_to_save if c in df_clean.columns]
    df_clean[cols_to_save].to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"  [OK] Saved -> {OUTPUT_FILE}")
    print(f"     Articles   : {len(df_clean)}")
    print(f"     Date range : {df_clean['Date'].min()} -> {df_clean['Date'].max()}")
    print(f"     Stocks     : {', '.join(successful)}")

    # FIX v4: per-stock score variance report
    score_summary = (
        df_clean.groupby("Stock")["news_score"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_score", "std": "std_score", "count": "n"})
        .sort_values("std_score")
    )
    print("\n  Per-stock news_score summary (sorted by std asc):")
    print(score_summary.to_string())

    low_var = score_summary[score_summary["std_score"] < NEWS_SCORE_MIN_STD]
    if not low_var.empty:
        print(f"\n  [WARNING] {len(low_var)} stock(s) have near-constant "
              f"news_score (std < {NEWS_SCORE_MIN_STD}):")
        print(low_var.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()