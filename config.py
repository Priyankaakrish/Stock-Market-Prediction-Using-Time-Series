"""
config.py — Central configuration for NVDA Stock Price Prediction Pipeline
Industry-standard approach: all tunable parameters in one place.
"""

import os
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import pathname2url

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_PROC_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models_saved"
LOGS_DIR      = BASE_DIR / "logs"
MLRUNS_DIR    = BASE_DIR / "mlruns"

for _d in [DATA_PROC_DIR, MODELS_DIR, LOGS_DIR, MLRUNS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Target Stock ──────────────────────────────────────────────────────────────
TICKER        = "NVDA"          # NVIDIA
TARGET_COL    = "Close"       # Column we forecast
FORECAST_DAYS = 30            # How many business days ahead to predict

# ── Data Sources ──────────────────────────────────────────────────────────────
DATA_FILES = {
    "master":      DATA_RAW_DIR / "nvda_master_dataset.csv",
    "yahoo":       DATA_RAW_DIR / "nvda_yahoo_finance.csv",
    # Optional extra sources — drop CSVs in data/raw/ and uncomment to merge:
    # "barchart":    DATA_RAW_DIR / "nvda_barchart.csv",
    # "investing":   DATA_RAW_DIR / "nvda_investing_com.csv",
    # "marketwatch": DATA_RAW_DIR / "nvda_marketwatch.csv",
    # "nasdaq":      DATA_RAW_DIR / "nvda_nasdaq.csv",
}

# ── Train / Validation / Test Split ───────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # must sum to 1.0

# ── Feature Engineering ───────────────────────────────────────────────────────
LAG_DAYS         = [1, 2, 3, 5, 10, 21]   # lag features
ROLLING_WINDOWS  = [5, 10, 20, 50, 200]   # MA / rolling-std windows
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BB_WINDOW        = 20
BB_STD           = 2
ATR_PERIOD       = 14

# ── ARIMA / SARIMA ────────────────────────────────────────────────────────────
ARIMA_MAX_P      = 3        # reduced from 5 — keeps search fast
ARIMA_MAX_D      = 1        # stock prices need at most 1 difference
ARIMA_MAX_Q      = 3        # reduced from 5
SARIMA_S         = 0        # 0 = disable seasonal search (negligible on stocks)
ARIMA_IC         = "aic"
ARIMA_SUBSAMPLE  = 500      # use only the last N rows to fit ARIMA
                            # ARIMA is short-memory; 500 rows >> needed; full
                            # 4k-row fit takes 20-40 min with no accuracy gain.

# ── Prophet ───────────────────────────────────────────────────────────────────
PROPHET_CHANGEPOINT_PRIOR   = 0.05
PROPHET_SEASONALITY_PRIOR   = 10.0
PROPHET_HOLIDAYS_PRIOR      = 10.0
PROPHET_YEARLY_SEASONALITY  = True
PROPHET_WEEKLY_SEASONALITY  = True
PROPHET_CV_HORIZON          = "90 days"
PROPHET_CV_INITIAL          = "730 days"
PROPHET_CV_PERIOD           = "180 days"

# ── LSTM ──────────────────────────────────────────────────────────────────────
LSTM_LOOKBACK       = 60       # sequence length (trading days)
LSTM_UNITS          = [128, 64]
LSTM_DROPOUT        = 0.2
LSTM_RECURRENT_DROP = 0.2
LSTM_DENSE_UNITS    = 32
LSTM_BATCH_SIZE     = 32
LSTM_EPOCHS         = 100
LSTM_PATIENCE       = 15       # early stopping patience
LSTM_LR             = 1e-3

# ── XGBoost / LightGBM ────────────────────────────────────────────────────────
XGB_N_TRIALS        = 50       # Optuna trials
XGB_CV_FOLDS        = 5
XGB_EARLY_STOP      = 50
LGBM_N_TRIALS       = 50

# ── Ensemble ──────────────────────────────────────────────────────────────────
# weights are optimised on validation set; these are fallback defaults
ENSEMBLE_WEIGHTS = {
    "arima":   0.15,
    "prophet": 0.20,
    "lstm":    0.35,
    "xgboost": 0.30,
}

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "NVDA_Stock_Prediction"
# On Windows, MLflow requires a file:// URI — bare C:\... paths are rejected.
# pathname2url converts backslashes and adds the correct leading slash.
MLFLOW_TRACKING_URI = "file:///" + pathname2url(str(MLRUNS_DIR)).lstrip("/")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 1))
API_RELOAD  = os.getenv("API_RELOAD", "false").lower() == "true"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
