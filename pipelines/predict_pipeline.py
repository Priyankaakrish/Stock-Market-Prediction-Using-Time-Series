"""
pipelines/predict_pipeline.py
──────────────────────────────
Loads saved model artefacts and produces a forward forecast
for the next FORECAST_DAYS business days.

Run:
    cd nvda_stock_prediction
    python pipelines/predict_pipeline.py [--steps 30] [--output forecast.csv]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.data.loader import load_all_sources
from src.data.preprocessor import clean, DataScaler
from src.forecasting import FEATURES_FILE, forecast_all, load_feature_list
from src.models.arima_model import ARIMAForecaster
from src.models.ensemble import EnsembleForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import GBForecaster

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format=cfg.LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_pipeline")


def load_models() -> dict:
    """Load all available saved models. Missing models are skipped gracefully."""
    models = {}
    loaders = [
        ("arima",   lambda: ARIMAForecaster.load(cfg.MODELS_DIR / "arima_model.pkl"),
                    cfg.MODELS_DIR / "arima_model.pkl"),
        ("prophet", lambda: ProphetForecaster.load(cfg.MODELS_DIR / "prophet_model.json"),
                    cfg.MODELS_DIR / "prophet_model.json"),
        ("lstm",    lambda: LSTMForecaster.load(),
                    cfg.MODELS_DIR / "lstm_model.pt"),
        ("xgboost", lambda: GBForecaster.load(cfg.MODELS_DIR / "xgboost_model.pkl",
                                              model_type="xgboost"),
                    cfg.MODELS_DIR / "xgboost_model.pkl"),
    ]
    for name, load, path in loaders:
        if path.exists():
            try:
                models[name] = load()
                logger.info("%s model loaded.", name)
            except Exception as e:
                logger.warning("Could not load %s: %s", name, e)

    if not models:
        raise RuntimeError(
            "No trained models found in models_saved/. "
            "Run train_pipeline.py first."
        )
    return models


def run_forecast(steps: int = cfg.FORECAST_DAYS) -> pd.DataFrame:
    """
    Full predict pipeline:
      1. Load latest data
      2. Run each model forward (recursively for LSTM / XGBoost)
      3. Blend with saved ensemble weights
      4. Return dated forecast DataFrame
    """
    logger.info("Loading latest NVDA data …")
    ohlcv = clean(load_all_sources())

    scaler = DataScaler("standard")
    scaler_path = cfg.MODELS_DIR / "feature_scaler.pkl"
    if scaler_path.exists():
        scaler.load(scaler_path)
    feat_cols = load_feature_list() if FEATURES_FILE.exists() else None

    models  = load_models()
    preds_d = forecast_all(models, ohlcv, steps=steps, scaler=scaler,
                           feat_cols=feat_cols)
    if not preds_d:
        raise RuntimeError("All model forecasts failed.")

    last_date    = ohlcv.index.max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=steps)

    ens_path = cfg.MODELS_DIR / "ensemble_weights.pkl"
    ens = EnsembleForecaster.load(ens_path) if ens_path.exists() else EnsembleForecaster()
    ens_df = ens.predict(preds_d)

    out = pd.DataFrame(preds_d, index=future_dates)
    out["ensemble"] = ens_df["ensemble"].values
    if "lower" in ens_df.columns:
        out["ci_lower"] = ens_df["lower"].values
        out["ci_upper"] = ens_df["upper"].values
    out.index.name = "Date"

    logger.info("Last close %.2f on %s", ohlcv[cfg.TARGET_COL].iloc[-1], last_date.date())
    logger.info("\nForecast for next %d business days:\n%s", steps, out.round(2).to_string())
    return out


def main():
    parser = argparse.ArgumentParser(
        description="NVDA Stock Prediction — Forward Forecast"
    )
    parser.add_argument("--steps",  type=int,  default=cfg.FORECAST_DAYS,
                        help="Number of business days to forecast")
    parser.add_argument("--output", type=str,  default="forecast_output.csv",
                        help="Output CSV filename (saved in data/processed/)")
    args = parser.parse_args()

    forecast = run_forecast(steps=args.steps)

    out_path = cfg.DATA_PROC_DIR / args.output
    forecast.to_csv(out_path)
    logger.info("Forecast saved -> %s", out_path)


if __name__ == "__main__":
    main()
