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
    "EMA_20", "RSI", "MACD", "ATR", "OBV", "Return_1d", "Direction",
]

FUNDAMENTAL_FY_PREFIXES = ["PE_Ratio_FY", "EPS_FY", "ROE_FY", "Debt_to_Equity_FY"]
FUNDAMENTAL_BASE_COLS   = ["PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
                            "Revenue_Growth", "Profit_Growth"]

NEWS_COLS = ["Date", "Stock", "News_Text", "Source"]

EVENTS_COLS = [
    "date", "symbol", "event_category", "event_name",
    "event_score_max", "event_count", "is_event",
]

# ── Labels ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
LABEL_MAP        = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV    = {0: "DOWN", 1: "FLAT", 2: "UP"}
DIRECTION_LABELS = ["DOWN", "FLAT", "UP"]

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.02,
    "subsample"       : 0.7,
    "colsample_bytree": 0.7,
    "gamma"           : 0.1,
    "reg_alpha"       : 0.1,
    "reg_lambda"      : 1.0,
    "eval_metric"     : "mlogloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
}

XGBOOST_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA_20", "RSI", "MACD", "ATR", "OBV", "Return_1d",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5",
    "RSI_lag1", "MACD_lag1", "OBV_lag1", "Return_1d_lag1",
    "Close_roll_mean_5", "Close_roll_std_5",
    "Close_roll_mean_10", "Close_roll_std_10",
    "Close_roll_mean_20",
    "PE_Ratio", "EPS", "ROE", "Debt_to_Equity",
    "Revenue_Growth", "Profit_Growth",
    "event_score_max", "event_count", "is_event",
    "news_positive", "news_neutral", "news_negative",
]

# ── LSTM ──────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
LSTM_FEATURES   = ["Close", "RSI", "MACD", "OBV", "ATR"]
LSTM_HIDDEN     = 64
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.3
LSTM_EPOCHS     = 50
LSTM_BATCH      = 64
LSTM_LR         = 0.0005
LSTM_PATIENCE   = 10

# ── FinBERT ───────────────────────────────────────────────────────────────────
FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_MAX_LEN    = 128
FINBERT_BATCH_SIZE = 16

# ── Auto-create directories ───────────────────────────────────────────────────
for _d in [
    DATA_DIR,
    os.path.join(DATA_DIR, "technical"),
    os.path.join(DATA_DIR, "fundamental"),
    os.path.join(DATA_DIR, "news"),
    os.path.join(DATA_DIR, "events"),
    os.path.join(DATA_DIR, "merged"),
    os.path.join(MODELS_DIR, "xgboost",  "saved"),
    os.path.join(MODELS_DIR, "lstm",     "saved"),
    os.path.join(MODELS_DIR, "ensemble", "saved"),
    os.path.join(EVAL_DIR,   "results"),
]:
    os.makedirs(_d, exist_ok=True)
