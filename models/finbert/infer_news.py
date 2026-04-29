"""
models/finbert/infer_news.py
============================
Lightweight FinBERT inference script.

Reads  data/news/news.csv  (columns: Date, Stock, News_Text, …)
Writes data/news/finbert_scores.csv with columns:
    Date, Stock, finbert_pos, finbert_neu, finbert_neg

The three probability columns are later used by train_meta.py as additional
inputs to the meta-learner.

Notes:
  • Uses ProsusAI/finbert from HuggingFace (downloads automatically on first run).
  • Batches headlines for GPU efficiency; falls back to CPU gracefully.
  • If a ticker already has FinBERT scores in the output file, those rows are
    skipped (incremental mode) — useful when re-running after partial failures.
  • Ticker normalisation is applied so M&M / M_M / M&M.NS all resolve to "M&M".
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

from config.settings import NEWS_CSV, FINBERT_MODEL, FINBERT_MAX_LEN, FINBERT_BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("finbert_infer")

# Output path sits alongside news.csv
FINBERT_OUTPUT = os.path.join(os.path.dirname(NEWS_CSV), "finbert_scores.csv")

# ── Ticker normalisation (same mapping as merge_features.py) ──────────────────
TICKER_ALIAS: dict[str, str] = {
    "M&M": "M&M", "M_M": "M&M", "M-M": "M&M", "MM": "M&M",
    "M&M.NS": "M&M", "M_M.NS": "M&M",
    "BAJAJ-AUTO": "BAJAJ-AUTO", "BAJAJ_AUTO": "BAJAJ-AUTO",
    "BAJAJAUTO": "BAJAJ-AUTO", "BAJAJ AUTO": "BAJAJ-AUTO",
    "BAJAJ-AUTO.NS": "BAJAJ-AUTO",
    "HDFC BANK": "HDFCBANK", "HDFC-BANK": "HDFCBANK",
}


def normalise_ticker(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.(NS|BO)$", "", regex=True).str.strip()
    return s.map(lambda t: TICKER_ALIAS.get(t, t))


# ── FinBERT label order (from ProsusAI/finbert config) ───────────────────────
# id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
_LABEL_ORDER = ["finbert_pos", "finbert_neg", "finbert_neu"]


def load_model(device):
    log.info(f"Loading {FINBERT_MODEL} ...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    model     = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval().to(device)
    log.info("FinBERT loaded ✓")
    return tokenizer, model


def infer_batch(texts: list[str], tokenizer, model, device,
                max_len: int = FINBERT_MAX_LEN) -> np.ndarray:
    """
    Returns ndarray of shape (N, 3): columns = [pos, neg, neu].
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits          # (N, 3)
    probs = softmax(logits, dim=-1).cpu().numpy()
    # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
    return probs   # columns: pos, neg, neu  (in model output order)


def run():
    if not os.path.exists(NEWS_CSV):
        raise FileNotFoundError(f"Missing {NEWS_CSV}. Run build_news.py first.")

    news = pd.read_csv(NEWS_CSV)
    news["Date"]  = pd.to_datetime(news["Date"])
    news["Stock"] = normalise_ticker(news["Stock"])

    if "News_Text" not in news.columns:
        raise ValueError("news.csv must have a 'News_Text' column.")

    # Fill missing texts with empty string (model handles short inputs fine)
    news["News_Text"] = news["News_Text"].fillna("").astype(str)

    log.info(f"Loaded {len(news):,} news rows  "
             f"({news['Stock'].nunique()} tickers)")

    # ── Incremental: skip already-processed rows ───────────────────────────
    existing = pd.DataFrame()
    if os.path.exists(FINBERT_OUTPUT):
        existing = pd.read_csv(FINBERT_OUTPUT, parse_dates=["Date"])
        log.info(f"Found existing scores: {len(existing):,} rows – running in incremental mode")
        done_keys = set(zip(existing["Date"].astype(str), existing["Stock"]))
        news["_key"] = list(zip(news["Date"].astype(str), news["Stock"]))
        news = news[~news["_key"].isin(done_keys)].drop(columns=["_key"])
        log.info(f"Rows to process: {len(news):,}")

    if news.empty:
        log.info("Nothing new to process — FinBERT scores are up to date.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Inference device: {device}")

    tokenizer, model = load_model(device)

    bs     = FINBERT_BATCH_SIZE
    texts  = news["News_Text"].tolist()
    n      = len(texts)
    all_probs = []

    for start in range(0, n, bs):
        batch = texts[start: start + bs]
        probs = infer_batch(batch, tokenizer, model, device)
        all_probs.append(probs)
        if (start // bs) % 10 == 0:
            log.info(f"  Processed {min(start + bs, n):,}/{n:,} headlines ...")

    all_probs = np.concatenate(all_probs, axis=0)   # (N, 3): pos, neg, neu

    result = news[["Date", "Stock"]].copy().reset_index(drop=True)
    result["finbert_pos"] = all_probs[:, 0]
    result["finbert_neg"] = all_probs[:, 1]
    result["finbert_neu"] = all_probs[:, 2]

    # Aggregate multiple headlines per (Date, Stock) → mean probabilities
    result = (
        result
        .groupby(["Date", "Stock"], as_index=False)
        [["finbert_pos", "finbert_neg", "finbert_neu"]]
        .mean()
    )

    # Append to existing scores
    if not existing.empty:
        result = pd.concat([existing, result], ignore_index=True)
        result = result.drop_duplicates(subset=["Date", "Stock"], keep="last")
        result = result.sort_values(["Stock", "Date"])

    os.makedirs(os.path.dirname(FINBERT_OUTPUT), exist_ok=True)
    result.to_csv(FINBERT_OUTPUT, index=False)
    log.info(f"FinBERT scores saved -> {FINBERT_OUTPUT}  ({len(result):,} rows)")

    # Quick sanity check
    log.info(f"  Mean pos={result['finbert_pos'].mean():.3f}  "
             f"neg={result['finbert_neg'].mean():.3f}  "
             f"neu={result['finbert_neu'].mean():.3f}")


if __name__ == "__main__":
    run()
