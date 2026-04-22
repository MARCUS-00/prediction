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
FUNDAMENTAL_FY_PREFIXES = ["PE_Ratio","EPS","ROE","Debt_to_Equity"]
NEWS_COLS   = ["Date","Stock","News_Text","Source"]
EVENTS_COLS = ["date","symbol","event_category","event_name",
               "event_score_max","event_count","is_event"]

# ── Labels ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
LABEL_MAP        = {-1: 0, 1: 1}
LABEL_MAP_INV    = {0: "DOWN", 1: "UP"}
DIRECTION_LABELS = ["DOWN", "UP"]

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"    : 800,
    "max_depth"       : 5,
    "learning_rate"   : 0.01,
    "subsample"       : 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "gamma"           : 0.2,
    "reg_alpha"       : 0.5,
    "reg_lambda"      : 2.0,
    "eval_metric"     : "logloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
}

XGBOOST_FEATURES = [
    # OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Base indicators
    "EMA_20", "RSI", "MACD", "MACD_signal", "ATR", "OBV", "Return_1d",
    # Lags
    "Close_lag1","Close_lag2","Close_lag3","Close_lag5",
    "RSI_lag1","MACD_lag1","OBV_lag1","Return_1d_lag1",
    # Rolling
    "Close_roll_mean_5","Close_roll_std_5",
    "Close_roll_mean_10","Close_roll_std_10",
    "Close_roll_mean_20","Close_roll_std_20",
    # Derived technical
    "BB_pct","Volume_ratio","Momentum_5d","Momentum_10d",
    "EMA_dist","RSI_overbought","RSI_oversold",
    # Cross-sectional (highest-signal new features)
    "CS_momentum_rank","CS_volume_rank","CS_rsi_rank",
    # Regime & context
    "streak_lag1","pct_from_52w_high",
    "market_vol_20d","intraday_range","gap_pct",
    # Fundamental
    "PE_Ratio","EPS","ROE","Debt_to_Equity",
    "Revenue","Profit","Revenue_Growth","Profit_Growth",
    # Events
    "event_score_max","event_count","is_event",
    # News
    "news_score","news_positive","news_negative",
]

# ── LSTM ──────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
LSTM_FEATURES   = [
    "Close","RSI","MACD","OBV","ATR",
    "Return_1d","EMA_dist","BB_pct","Volume_ratio",
    "Momentum_5d","news_score",
    "CS_momentum_rank","CS_rsi_rank","market_vol_20d",
]
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 60
LSTM_BATCH      = 128
LSTM_LR         = 0.001
LSTM_PATIENCE   = 12

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