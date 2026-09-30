"""
api/main.py
────────────
FastAPI application entry point for the NVDA Stock Prediction service.

Startup:
  • Loads all trained model artefacts into app.state.model_registry
  • Loads the latest market data for context
  • Registers all routers

Run (development):
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Run (production via gunicorn):
    gunicorn api.main:app -k uvicorn.workers.UvicornWorker \
        -w 1 -b 0.0.0.0:8000

OpenAPI docs: http://localhost:8000/docs
Redoc:        http://localhost:8000/redoc
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from api.routers import predict as predict_router

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format=cfg.LOG_FORMAT,
)
logger = logging.getLogger("api.main")


# ── Model registry loader ─────────────────────────────────────────────────────

def _load_registry() -> dict:
    """
    Load the latest price history, the saved models, the LSTM feature scaler
    and the model feature list. Forecasts are computed per request from this.
    """
    from src.data.loader import load_all_sources
    from src.data.preprocessor import clean, DataScaler
    from src.forecasting import FEATURES_FILE, load_feature_list
    from src.models.arima_model import ARIMAForecaster
    from src.models.ensemble import EnsembleForecaster
    from src.models.lstm_model import LSTMForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.xgboost_model import GBForecaster

    registry: dict = {"models": {}}

    # ── Load latest data ──────────────────────────────────────────────────────
    try:
        ohlcv = clean(load_all_sources())
        registry["ohlcv"]            = ohlcv
        registry["last_data_date"]   = str(ohlcv.index.max().date())
        registry["last_date_series"] = ohlcv.index.max()
        logger.info("Data loaded: %d rows, last date %s",
                    len(ohlcv), registry["last_data_date"])
    except Exception as e:
        logger.error("Data loading failed: %s", e)

    scaler_path = cfg.MODELS_DIR / "feature_scaler.pkl"
    if scaler_path.exists():
        scaler = DataScaler("standard")
        scaler.load(scaler_path)
        registry["scaler"] = scaler
    if FEATURES_FILE.exists():
        registry["feature_cols"] = load_feature_list()

    # ── Load models ───────────────────────────────────────────────────────────
    loaders = [
        ("arima",   ARIMAForecaster.load,   cfg.MODELS_DIR / "arima_model.pkl"),
        ("prophet", ProphetForecaster.load, cfg.MODELS_DIR / "prophet_model.json"),
        ("lstm",    LSTMForecaster.load,    cfg.MODELS_DIR / "lstm_model.pt"),
        ("xgboost", lambda p: GBForecaster.load(p, "xgboost"),
                    cfg.MODELS_DIR / "xgboost_model.pkl"),
    ]

    for name, loader, path in loaders:
        if path.exists():
            try:
                registry["models"][name] = loader(path)
                logger.info("Loaded model: %s", name)
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ens_path = cfg.MODELS_DIR / "ensemble_weights.pkl"
    if ens_path.exists():
        try:
            registry["ensemble"] = EnsembleForecaster.load(ens_path)
            logger.info("Ensemble weights loaded.")
        except Exception as e:
            logger.warning("Failed to load ensemble: %s", e)

    logger.info("Registry ready: models=%s", list(registry["models"].keys()))
    return registry


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NVDA Stock Prediction API …")
    app.state.model_registry = _load_registry()
    logger.info("API ready.")
    yield
    logger.info("API shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="NVDA Stock Price Prediction API",
        description=(
            "Multi-model time-series forecasting service for NVIDIA (NVDA) stock.\n\n"
            "Models available: ARIMA/SARIMA · Prophet · Bidirectional LSTM · XGBoost.\n"
            "Ensemble blending with optimised weights."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handler ─────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(predict_router.router)

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "NVDA Stock Prediction API",
            "version": "1.0.0",
            "docs":    "/docs",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        reload=cfg.API_RELOAD,
        workers=cfg.API_WORKERS,
        log_level=cfg.LOG_LEVEL.lower(),
    )
