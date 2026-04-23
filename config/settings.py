import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Paths ─────────────────────────────────────────────────────────────────────
TECHNICAL_CSV   = os.path.join(DATA_DIR, "technical",   "technical.csv")
FUNDAMENTAL_CSV = os.path.join(DATA_DIR, "fundamental", "fundamental.csv")
NEWS_CSV        = os.path.join(DATA_DIR, "news",        "news.csv")
EVENTS_CSV      = os.path.join(DATA_DIR, "events",      "events.csv")
MERGED_CSV      = os.path.join(DATA_DIR, "merged",      "merged_final.csv")

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR       = os.path.join(BASE_DIR, "models")
XGB_MODEL_PATH   = os.path.join(MODELS_DIR, "xgboost",  "saved", "xgb_model.pkl")
LSTM_MODEL_PATH  = os.path.join(MODELS_DIR, "lstm",     "saved", "lstm_model.pt")
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "lstm",     "saved", "lstm_scaler.pkl")
META_MODEL_PATH  = os.path.join(MODELS_DIR, "ensemble", "saved", "meta_learner.pkl")

# ── Evaluation paths ──────────────────────────────────────────────────────────
EVAL_DIR              = os.path.join(BASE_DIR, "evaluation")
XGB_RESULTS_PATH      = os.path.join(EVAL_DIR, "results", "xgboost_results.csv")
LSTM_RESULTS_PATH     = os.path.join(EVAL_DIR, "results", "lstm_results.csv")
ENSEMBLE_RESULTS_PATH = os.path.join(EVAL_DIR, "results", "ensemble_results.csv")
WATCHLIST_OUTPUT_PATH = os.path.join(EVAL_DIR, "results", "watchlist_latest.csv")

# ── Column definitions ────────────────────────────────────────────────────────
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

# ── Sector map ────────────────────────────────────────────────────────────────
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

# ── Labels ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
LABEL_MAP        = {-1: 0, 1: 1}
LABEL_MAP_INV    = {0: "DOWN", 1: "UP"}
DIRECTION_LABELS = ["DOWN", "UP"]

# ── XGBoost ───────────────────────────────────────────────────────────────────
# FIX: removed scale_pos_weight (set dynamically in train.py based on class counts,
#      but using it with early_stopping causes DOWN bias when AUC~0.50)
# FIX: early_stopping_rounds removed here — passed in XGBClassifier() constructor
XGBOOST_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 3,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 10,
    "reg_alpha"       : 0.2,
    "reg_lambda"      : 1.5,
    "eval_metric"     : "logloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
    "tree_method"     : "hist",
}

# FIX: Revenue and Profit removed (raw rupee values, scale too large for trees
#      without log-transform; replaced by Revenue_Growth and Profit_Growth).
# FIX: OBV removed (raw OBV scale dominates; OBV_change is the normalised version).
# All features verified present in merged_final.csv.
XGBOOST_FEATURES = [
    # OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Base indicators
    "EMA_9", "EMA_20", "EMA_50", "RSI", "MACD", "MACD_signal", "MACD_hist",
    "ATR", "ATR_ratio", "Return_1d",
    # Lags
    "Close_lag1","Close_lag2","Close_lag3","Close_lag5",
    "RSI_lag1","MACD_lag1","Return_1d_lag1",
    # Rolling
    "Close_roll_mean_5","Close_roll_std_5",
    "Close_roll_mean_10","Close_roll_std_10",
    "Close_roll_mean_20","Close_roll_std_20",
    # Derived technical
    "BB_pct","Volume_ratio","volume_shock","Momentum_5d","Momentum_10d",
    "EMA_dist","EMA_dist_50","RSI_overbought","RSI_oversold",
    "RSI_change","OBV_change","price_accel","EMA_cross_9_20",
    # Cross-sectional
    "CS_momentum_rank","CS_volume_rank","CS_rsi_rank","CS_atr_rank",
    # Regime & context
    "pct_from_52w_high","pct_from_52w_low","sector_encoded",
    "market_vol_20d","intraday_range","gap_pct",
    # Fundamental (growth rates only — raw Revenue/Profit removed)
    "PE_Ratio","EPS","ROE","Debt_to_Equity",
    "Revenue_Growth","Profit_Growth",
    # Events
    "event_score_max","event_count","is_event",
    # News (news_score_5d and has_news kept but will be 0 if no news.csv)
    "news_score","news_score_5d","news_positive","news_negative","news_count","has_news",
]

# ── LSTM ──────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
LSTM_FEATURES   = [
    "Close","RSI","RSI_change","MACD","MACD_hist","ATR","ATR_ratio",
    "Return_1d","EMA_dist","BB_pct","Volume_ratio",
    "Momentum_5d","OBV_change","news_score","news_score_5d","has_news",
    "CS_momentum_rank","CS_rsi_rank","market_vol_20d",
]
LSTM_HIDDEN   = 128
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.3
LSTM_EPOCHS   = 60
LSTM_BATCH    = 128
LSTM_LR       = 0.001
LSTM_PATIENCE = 15   # FIX: increased from 12 to 15 (model was stopping too early)

# ── FinBERT ───────────────────────────────────────────────────────────────────
FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_MAX_LEN    = 128
FINBERT_BATCH_SIZE = 32

# ── Auto-create directories ───────────────────────────────────────────────────
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
