# =============================================================================
# config/settings.py  (FIXED v9)
#
# Root-cause fixes:
#   1. LABEL_THRESHOLD REDUCED 0.01 → 0.005 (0.5%)
#      The 1% threshold dropped 25% of rows AND created a bull-market test
#      period where UP% = 62%, causing XGBoost to collapse to majority=UP
#      and still hit 54% accuracy. 0.5% keeps ~87% of rows; class ratio
#      stays near 50/50 across time periods.
#
#   2. scale_pos_weight computed dynamically in train.py (placeholder here).
#      This is the missing piece that caused hard class collapse.
#
#   3. XGBoost: min_child_weight 15→8 (was over-regularized), max_depth 4→5.
#      Added return_vs_sector and news_rolling_3d features.
#      Removed raw price lag cols (Close_lag1/2/3/5).
#
#   4. LSTM: hidden 128→64, layers 2→1, LR 0.0005→0.001, patience 25→15.
#      A 2-layer 128-unit LSTM has ~800K params for 40 stocks × 2700 days —
#      it overfits severely. Val loss diverged by epoch 3.
# =============================================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TECHNICAL_CSV   = os.path.join(DATA_DIR, "technical",   "technical.csv")
FUNDAMENTAL_CSV = os.path.join(DATA_DIR, "fundamental", "fundamental.csv")
NEWS_CSV        = os.path.join(DATA_DIR, "news",        "news.csv")
EVENTS_CSV      = os.path.join(DATA_DIR, "events",      "events.csv")
MERGED_CSV      = os.path.join(DATA_DIR, "merged",      "merged_final.csv")

MODELS_DIR       = os.path.join(BASE_DIR, "models")
XGB_MODEL_PATH   = os.path.join(MODELS_DIR, "xgboost",  "saved", "xgb_model.pkl")
LSTM_MODEL_PATH  = os.path.join(MODELS_DIR, "lstm",     "saved", "lstm_model.pt")
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "lstm",     "saved", "lstm_scaler.pkl")
META_MODEL_PATH  = os.path.join(MODELS_DIR, "ensemble", "saved", "meta_learner.pkl")

EVAL_DIR              = os.path.join(BASE_DIR, "evaluation")
XGB_RESULTS_PATH      = os.path.join(EVAL_DIR, "results", "xgboost_results.csv")
LSTM_RESULTS_PATH     = os.path.join(EVAL_DIR, "results", "lstm_results.csv")
ENSEMBLE_RESULTS_PATH = os.path.join(EVAL_DIR, "results", "ensemble_results.csv")
WATCHLIST_OUTPUT_PATH = os.path.join(EVAL_DIR, "results", "watchlist_latest.csv")

TECHNICAL_COLS = [
    "Date", "Stock", "Open", "High", "Low", "Close", "Volume",
    "EMA_20", "RSI", "MACD", "MACD_signal", "ATR", "OBV",
    "Return_1d", "Direction",
]
FUNDAMENTAL_BASE_COLS = ["Year","PE_Ratio","EPS","ROE","Debt_to_Equity",
                          "Revenue","Profit","Revenue_Growth","Profit_Growth"]
NEWS_COLS   = ["Date","Stock","News_Text","Source"]
EVENTS_COLS = ["date","symbol","event_category","event_name",
               "event_score_max","event_count","is_event"]

SECTOR_MAP = {
    "HDFCBANK": "Financial_Services", "ICICIBANK": "Financial_Services",
    "SBIN": "Financial_Services",     "AXISBANK": "Financial_Services",
    "KOTAKBANK": "Financial_Services","BAJFINANCE": "Financial_Services",
    "BAJAJFINSV": "Financial_Services","INDUSINDBK": "Financial_Services",
    "TCS": "Information_Technology",  "INFY": "Information_Technology",
    "HCLTECH": "Information_Technology","WIPRO": "Information_Technology",
    "TECHM": "Information_Technology","RELIANCE": "Energy",
    "ONGC": "Energy",                 "NTPC": "Utilities",
    "POWERGRID": "Utilities",         "BPCL": "Energy",
    "HINDUNILVR": "Consumer_Staples", "ITC": "Consumer_Staples",
    "NESTLEIND": "Consumer_Staples",  "BRITANNIA": "Consumer_Staples",
    "MARUTI": "Automobile",           "M&M": "Automobile",
    "BHARTIARTL": "Telecom",          "EICHERMOT": "Automobile",
    "HEROMOTOCO": "Automobile",       "BAJAJ-AUTO": "Automobile",
    "SUNPHARMA": "Pharma",            "CIPLA": "Pharma",
    "DRREDDY": "Pharma",              "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",             "HINDALCO": "Metals",
    "COALINDIA": "Energy",            "LT": "Industrials",
    "ULTRACEMCO": "Cement",           "GRASIM": "Cement",
    "ASIANPAINT": "Consumer_Durables","TITAN": "Consumer_Durables",
}
SECTOR_NAMES   = sorted(set(SECTOR_MAP.values()))
SECTOR_TO_CODE = {name: i for i, name in enumerate(SECTOR_NAMES)}

# ─── LABEL STRATEGY (FIX v9) ─────────────────────────────────────────────────
LABEL_HORIZON    = 5       # 5-trading-day forward direction (keep)
LABEL_THRESHOLD  = 0.005   # FIX: was 0.01 → now 0.005 (keeps ~87% of rows)
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15

LABEL_MAP        = {-1: 0, 1: 1}
LABEL_MAP_INV    = {0: "DOWN", 1: "UP"}
DIRECTION_LABELS = ["DOWN", "UP"]

# ─── XGBOOST (FIX v9) ────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"    : 2000,
    "max_depth"       : 5,         # FIX: was 4
    "learning_rate"   : 0.01,
    "subsample"       : 0.75,
    "colsample_bytree": 0.6,
    "min_child_weight": 8,         # FIX: was 15 (too conservative)
    "reg_alpha"       : 0.5,
    "reg_lambda"      : 2.0,
    "eval_metric"     : "logloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
    "tree_method"     : "hist",
    # scale_pos_weight injected dynamically in train.py
}

XGBOOST_FEATURES = [
    # Core price momentum
    "Return_1d", "return_3d", "ret_5d", "ret_10d", "ret_20d",
    "Momentum_5d", "Momentum_10d", "norm_mom5", "norm_mom10",
    # Trend / EMA
    "EMA_dist", "EMA_dist_50", "EMA_cross_9_20", "price_above_ema200", "BB_pct",
    # RSI
    "RSI", "RSI_change", "rsi_momentum",
    # Volatility
    "ATR_ratio", "vol10d", "sharpe_5d", "volatility_5d",
    # Volume
    "Volume_ratio", "vol_ratio20", "OBV_change",
    # Intraday structure
    "hl_ratio", "close_range_pct", "gap_pct", "gap_close_pct",
    # Cross-sectional rank
    "CS_momentum_rank", "CS_rsi_rank", "CS_volume_rank",
    # NIFTY50 market context
    "nifty_ret_1d", "nifty_ret_5d", "nifty_rsi", "nifty_above_ema20",
    # Sector context + alpha (NEW)
    "sector_ret_1d", "sector_ret_5d", "sector_encoded", "return_vs_sector",
    # 52-week position
    "pct_from_52w_high", "pct_from_52w_low",
    # Fundamental
    "ROE", "Revenue_Growth", "Profit_Growth", "PE_Ratio_norm",
    # Events
    "event_score_max", "is_event",
    # News — rolling window smooths sparsity (NEW)
    "news_score", "news_rolling_3d", "news_spike", "has_news",
]

# ─── LSTM (FIX v9) ────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 15
LSTM_FEATURES   = [
    "Return_1d", "return_3d", "ret_5d", "ret_10d",
    "RSI", "RSI_change",
    "MACD", "MACD_hist",
    "ATR_ratio", "EMA_dist", "BB_pct",
    "Volume_ratio", "vol10d",
    "norm_mom5", "hl_ratio", "close_range_pct",
    "rsi_momentum",
    "CS_momentum_rank", "CS_rsi_rank",
    "nifty_ret_1d", "nifty_ret_5d", "nifty_above_ema20",
    "sector_ret_1d", "return_vs_sector",
    "news_score", "news_rolling_3d",
]

LSTM_HIDDEN   = 64    # FIX: was 128
LSTM_LAYERS   = 1     # FIX: was 2 (val loss diverged at epoch 3)
LSTM_DROPOUT  = 0.3
LSTM_EPOCHS   = 60
LSTM_BATCH    = 256
LSTM_LR       = 0.001  # FIX: was 0.0005
LSTM_PATIENCE = 15     # FIX: was 25

FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_MAX_LEN    = 128
FINBERT_BATCH_SIZE = 32

for _d in [
    DATA_DIR,
    os.path.join(DATA_DIR, "technical"), os.path.join(DATA_DIR, "fundamental"),
    os.path.join(DATA_DIR, "news"), os.path.join(DATA_DIR, "events"),
    os.path.join(DATA_DIR, "merged"),
    os.path.join(MODELS_DIR, "xgboost", "saved"),
    os.path.join(MODELS_DIR, "lstm",    "saved"),
    os.path.join(MODELS_DIR, "ensemble","saved"),
    os.path.join(EVAL_DIR, "results"),
]:
    os.makedirs(_d, exist_ok=True)
