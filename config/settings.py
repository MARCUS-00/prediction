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

DATE_START = "2020-01-01"
DATE_END   = "2025-12-31"

TRAIN_END  = "2023-12-31"
VAL_START  = "2024-01-01"
VAL_END    = "2024-12-31"
TEST_START = "2025-01-01"

LABEL_HORIZON   = 5
# FIX 4: Align LABEL_THRESHOLD with the 1.5% threshold actually used in
#         merge_features.py (UP_THRESH / DOWN_THRESH = ±0.015).
#         Old value 0.005 was inconsistent; FLAT class was ~43%, now ~30%.
LABEL_THRESHOLD = 0.015
RANDOM_SEED     = 42
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15

# ── 3-class label maps ────────────────────────────────────────────────────────
# External label space : {-1: DOWN, 0: FLAT,  1: UP}
# Internal label space : { 0: DOWN, 1: FLAT,  2: UP}  (used by XGB / LSTM)
# LABEL_MAP_INV maps INTERNAL → human string (matches internal index)
LABEL_MAP        = {-1: 0, 1: 2}           # kept for legacy callers; prefer EXT_TO_INT
LABEL_MAP_INV    = {0: "DOWN", 1: "FLAT", 2: "UP"}   # internal {0,1,2} → string
DIRECTION_LABELS = ["DOWN", "FLAT", "UP"]

# External → Internal and back (canonical; used by train.py, predict.py, etc.)
EXT_TO_INT = {-1: 0, 0: 1, 1: 2}
INT_TO_EXT = {0: -1, 1: 0, 2: 1}

SECTOR_MAP = {
    "HDFCBANK":   "Financial_Services", "ICICIBANK":  "Financial_Services",
    "SBIN":       "Financial_Services", "AXISBANK":   "Financial_Services",
    "KOTAKBANK":  "Financial_Services", "BAJFINANCE": "Financial_Services",
    "BAJAJFINSV": "Financial_Services", "INDUSINDBK": "Financial_Services",
    "TCS":        "Information_Technology", "INFY":   "Information_Technology",
    "HCLTECH":    "Information_Technology", "WIPRO":  "Information_Technology",
    "TECHM":      "Information_Technology", "RELIANCE": "Energy",
    "ONGC":       "Energy",            "NTPC":       "Utilities",
    "POWERGRID":  "Utilities",         "BPCL":       "Energy",
    "HINDUNILVR": "Consumer_Staples",  "ITC":        "Consumer_Staples",
    "NESTLEIND":  "Consumer_Staples",  "BRITANNIA":  "Consumer_Staples",
    "MARUTI":     "Automobile",        "M&M":        "Automobile",
    "BHARTIARTL": "Telecom",           "EICHERMOT":  "Automobile",
    "HEROMOTOCO": "Automobile",        "BAJAJ-AUTO": "Automobile",
    "SUNPHARMA":  "Pharma",            "CIPLA":      "Pharma",
    "DRREDDY":    "Pharma",            "TATASTEEL":  "Metals",
    "JSWSTEEL":   "Metals",            "HINDALCO":   "Metals",
    "COALINDIA":  "Energy",            "LT":         "Industrials",
    "ULTRACEMCO": "Cement",            "GRASIM":     "Cement",
    "ASIANPAINT": "Consumer_Durables", "TITAN":      "Consumer_Durables",
}
SECTOR_NAMES   = sorted(set(SECTOR_MAP.values()))
SECTOR_TO_CODE = {name: i for i, name in enumerate(SECTOR_NAMES)}

XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        5,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.2,
    "reg_alpha":        0.5,
    "reg_lambda":       2.0,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
}

XGBOOST_FEATURES = [
    "Return_1d",
    # FIX 7: Add lag features — strictly no look-ahead (shift≥1 applied in merge_features)
    "ret_lag_1d", "ret_lag_3d", "ret_lag_5d",
    "momentum_5d", "momentum_10d", "momentum_diff", "momentum_strength",
    "RSI", "EMA_20",
    "MACD_hist", "MACD_signal",
    "ATR", "OBV",
    "volatility_ratio",
    "vol_spike", "vol_breakout",
    "price_pos_20d",
    "nifty_ret_1d", "nifty_ret_5d",
    "ret_vs_nifty_1d", "ret_vs_nifty_5d",
    "sector_ret_1d", "sector_ret_5d", "return_vs_sector", "sector_encoded",
    "PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth",
    # FIX 3: has_fundamental_data flag so model knows when fundamentals are real vs filled
    "has_fundamental_data",
    "news_score", "news_rolling_3d", "news_decay", "has_news",
    "event_score_max", "is_event", "event_strength", "event_impact_decay",
    "alpha_strength",
]

SEQUENCE_LENGTH = 15
LSTM_FEATURES = [
    "Return_1d",
    # FIX 7: Add lag features to LSTM input
    "ret_lag_1d", "ret_lag_3d", "ret_lag_5d",
    "momentum_5d", "momentum_10d", "momentum_diff",
    "RSI", "MACD_hist", "MACD_signal",
    "volatility_ratio", "ATR",
    "vol_spike", "price_pos_20d",
    "nifty_ret_1d", "nifty_ret_5d",
    "ret_vs_nifty_1d", "ret_vs_nifty_5d",
    "sector_ret_1d", "return_vs_sector",
    "news_score", "news_rolling_3d",
    "event_strength",
    "has_fundamental_data",
]

# ── LSTM hyper-parameters (single source of truth) ───────────────────────────
LSTM_HIDDEN   = 64
# FIX 6: Set LSTM_LAYERS=2. With 1 layer PyTorch silently ignores the
#         dropout parameter inside the LSTM cell. 2 layers activates it.
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.3
LSTM_EPOCHS   = 60
LSTM_BATCH    = 256
LSTM_LR       = 0.001
LSTM_PATIENCE = 15

FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_MAX_LEN    = 128
FINBERT_BATCH_SIZE = 32

for _d in [
    DATA_DIR,
    os.path.join(DATA_DIR, "technical"), os.path.join(DATA_DIR, "fundamental"),
    os.path.join(DATA_DIR, "news"), os.path.join(DATA_DIR, "events"),
    os.path.join(DATA_DIR, "merged"),
    os.path.join(MODELS_DIR, "xgboost",  "saved"),
    os.path.join(MODELS_DIR, "lstm",     "saved"),
    os.path.join(MODELS_DIR, "ensemble", "saved"),
    os.path.join(EVAL_DIR,   "results"),
]:
    os.makedirs(_d, exist_ok=True)
