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
# FIX: Threshold-based labeling — only confident directional moves are kept.
# return > +0.5% → 1 (UP), return < -0.5% → 0 (DOWN), otherwise → dropped.
# Removes ambiguous "noise" rows and makes the task learnable.
LABEL_THRESHOLD  = 0.005   # 0.5%
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15

# Direction column in merged CSV stays -1/+1.
# Threshold filtering is applied at training time in each train.py.
# LABEL_MAP converts filtered rows: 1->1, -1->0
LABEL_MAP        = {-1: 0, 1: 1}
LABEL_MAP_INV    = {0: "DOWN", 1: "UP"}
DIRECTION_LABELS = ["DOWN", "UP"]

# ── XGBoost ───────────────────────────────────────────────────────────────────
# FIX: Increased capacity (n_estimators 300→500, max_depth 3→6, lr 0.05→0.03)
# FIX: colsample_bytree 0.6→0.8 — was starving trees of features
# FIX: min_child_weight 10→5 — was too conservative
# FIX: Relaxed regularization slightly
XGBOOST_PARAMS = {
    "n_estimators"    : 500,
    "max_depth"       : 6,
    "learning_rate"   : 0.03,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "eval_metric"     : "logloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
    "tree_method"     : "hist",
}

# Revenue and Profit removed (raw rupee values, not normalised).
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
    "BB_pct","Volume_ratio","volume_shock","volume_change","Momentum_5d","Momentum_10d",
    "EMA_dist","EMA_dist_50","RSI_overbought","RSI_oversold",
    "RSI_change","OBV_change","price_accel","EMA_cross_9_20","return_3d","volatility_5d","price_above_ema200",
    # Cross-sectional
    "CS_momentum_rank","CS_volume_rank","CS_rsi_rank","CS_atr_rank",
    # Regime & context
    "pct_from_52w_high","pct_from_52w_low","sector_encoded",
    "market_vol_20d","intraday_range","gap_pct",
    # Fundamental (growth rates only)
    "PE_Ratio","EPS","ROE","Debt_to_Equity",
    "Revenue_Growth","Profit_Growth",
    # Events
    "event_score_max","event_count","is_event",
    # News
    "news_score","news_score_5d","news_score_7d","news_positive","news_negative","news_count","has_news",
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
LSTM_PATIENCE = 15

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
