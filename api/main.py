"""FastAPI inference service for NVDA next-day close forecasts.

Endpoints
---------
GET  /health    liveness + model/data status (used by Docker and the ALB)
GET  /model     metadata and held-out metrics for the deployed model
POST /forecast  next-day close, optionally against caller-supplied bars
GET  /metrics   Prometheus-style counters

The service reuses the *exact* preprocessing and feature-engineering code from
``src`` rather than reimplementing it. Training/serving skew in a technical-
indicator pipeline is silent and brutal: an RSI computed with a slightly
different smoothing rule shifts every prediction without raising an error.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src import __version__
from src.config import DATA, PATHS
from src.features import build_features
from src.models.base import Forecaster
from src.preprocess import preprocess

from .monitoring import MONITOR, drift_summary
from .schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
)

log = logging.getLogger("api")

DISCLAIMER = (
    "Model output for research and educational use only. This is not "
    "investment advice, and a next-day price forecast should not be used to "
    "make trading decisions. See the README for held-out performance versus a "
    "random-walk baseline."
)

# Minimum bars needed for the longest rolling window (MA50, 60-day z-score)
# plus a safety margin.
MIN_BARS = 120

STATE: dict = {"model": None, "meta": {}, "data": None, "loaded_at": None}
COUNTERS = {"forecast_requests": 0, "forecast_errors": 0, "latency_sum_ms": 0.0}


def _load_model() -> None:
    """Load from the MLflow registry if configured, else the local pickle.

    The two sources return different objects: the registry hands back a
    ``PyFuncModel`` exposing only ``predict``, while the pickle is a
    ``Forecaster`` exposing ``backtest``. ``STATE["kind"]`` records which, so
    the forecast handler dispatches instead of assuming — the earlier version
    assumed the pickle and returned 500 on every request in registry mode.
    """
    # Metadata first: it describes the champion regardless of where the model
    # is loaded from, and the registry branch used to return before reaching it,
    # leaving /health and /model reporting nulls.
    meta_path = PATHS.models / "best_model_meta.json"
    if meta_path.exists():
        STATE["meta"] = json.loads(meta_path.read_text())

    uri = os.getenv("MODEL_URI")
    if uri:
        try:
            import mlflow.pyfunc

            tracking = os.getenv("MLFLOW_TRACKING_URI")
            if tracking:
                mlflow.set_tracking_uri(tracking)
            STATE["model"] = mlflow.pyfunc.load_model(uri)
            STATE["kind"] = "pyfunc"
            STATE["source"] = uri
            log.info("Loaded model from registry: %s", uri)
            return
        except Exception as err:
            log.warning("Registry load failed (%s); falling back to pickle", err)

    if PATHS.best_model.exists():
        STATE["model"] = Forecaster.load(PATHS.best_model)
        STATE["kind"] = "forecaster"
        STATE["source"] = str(PATHS.best_model)
        log.info("Loaded model from %s", PATHS.best_model)
    else:
        log.error("No model artefact found at %s — run `python -m src.train`",
                  PATHS.best_model)


def _predict_next(feats: pd.DataFrame) -> tuple[float, str]:
    """Next-day close from whichever model kind is loaded.

    ``feats`` ends with the row to forecast from, preceded by enough history
    for the model's lookback. Returns (predicted_close, model_name).
    """
    model = STATE["model"]

    if STATE.get("kind") == "pyfunc":
        # The pyfunc wrapper takes the whole frame and splits history/target
        # itself, so the contract here is "hand it the window".
        frame = feats.drop(columns=["target", "target_return"], errors="ignore")
        out = model.predict(frame)
        return float(out["predicted_close"].iloc[0]), str(out["model"].iloc[0])

    history, latest = feats.iloc[:-1], feats.iloc[-1:]
    name = STATE.get("meta", {}).get("model", getattr(model, "name", "unknown"))
    return float(model.backtest(history, latest)[0]), name


def _load_data() -> None:
    if PATHS.cleaned.exists():
        df = pd.read_csv(PATHS.cleaned, parse_dates=["Date"])
        STATE["data"] = df
        log.info("Loaded %d cached bars (through %s)", len(df), df["Date"].max().date())


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _load_model()
    _load_data()
    STATE["loaded_at"] = time.time()
    yield
    STATE.clear()


app = FastAPI(
    title="NVDA Stock Price Forecasting API",
    description="Next-day close forecasts for NVIDIA (NVDA) from a "
                "time-series ML pipeline.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    data = STATE.get("data")
    return HealthResponse(
        status="ok" if STATE.get("model") is not None else "degraded",
        model_loaded=STATE.get("model") is not None,
        model_name=STATE.get("meta", {}).get("model"),
        trained_at=STATE.get("meta", {}).get("trained_at"),
        data_rows=0 if data is None else len(data),
        data_last_date=None if data is None else str(data["Date"].max().date()),
        version=__version__,
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    meta = STATE.get("meta", {})
    if not meta:
        raise HTTPException(503, "Model metadata unavailable; has training run?")
    return ModelInfoResponse(
        model_name=meta.get("model", "unknown"),
        trained_at=meta.get("trained_at"),
        horizon_days=meta.get("horizon_days", DATA.horizon),
        train_end=meta.get("train_end"),
        test_window=f"{meta.get('test_start')} to {meta.get('test_end')}",
        test_metrics=meta.get("metrics", {}),
        naive_rmse=meta.get("naive_rmse"),
        mlflow_run_id=meta.get("mlflow_run_id"),
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    n = max(COUNTERS["forecast_requests"], 1)
    lines = [
        "# HELP nvda_forecast_requests_total Forecast requests served.",
        "# TYPE nvda_forecast_requests_total counter",
        f"nvda_forecast_requests_total {COUNTERS['forecast_requests']}",
        "# HELP nvda_forecast_errors_total Failed forecast requests.",
        "# TYPE nvda_forecast_errors_total counter",
        f"nvda_forecast_errors_total {COUNTERS['forecast_errors']}",
        "# HELP nvda_forecast_latency_ms_avg Mean forecast latency.",
        "# TYPE nvda_forecast_latency_ms_avg gauge",
        f"nvda_forecast_latency_ms_avg {COUNTERS['latency_sum_ms'] / n:.3f}",
        "# HELP nvda_model_loaded Whether a model artefact is in memory.",
        "# TYPE nvda_model_loaded gauge",
        f"nvda_model_loaded {int(STATE.get('model') is not None)}",
    ]
    lines += MONITOR.prometheus_lines()
    return "\n".join(lines) + "\n"


@app.post("/forecast", response_model=ForecastResponse, tags=["inference"])
def forecast(req: ForecastRequest) -> ForecastResponse:
    started = time.perf_counter()
    COUNTERS["forecast_requests"] += 1

    model = STATE.get("model")
    if model is None:
        COUNTERS["forecast_errors"] += 1
        raise HTTPException(503, "No model loaded. Run `python -m src.train` first.")

    try:
        if req.bars:
            if len(req.bars) < MIN_BARS:
                raise HTTPException(
                    422, f"Need at least {MIN_BARS} bars to warm up the technical "
                         f"indicators; received {len(req.bars)}.")
            raw = pd.DataFrame([
                {"Date": b.date, "Open": b.open, "High": b.high, "Low": b.low,
                 "Close": b.close, "Volume": b.volume}
                for b in req.bars
            ])
            raw["Date"] = pd.to_datetime(raw["Date"])
            if not raw["Date"].is_monotonic_increasing:
                raise HTTPException(422, "bars must be in ascending date order")
            cleaned = preprocess(raw, save=False)
        else:
            cleaned = STATE.get("data")
            if cleaned is None:
                raise HTTPException(503, "No cached history available on the server")

        # Feature building drops the final row (no label). Append a duplicate
        # sentinel row so the true last bar survives, then discard the sentinel.
        padded = pd.concat([cleaned, cleaned.tail(1)], ignore_index=True)
        feats = build_features(padded, save=False)
        if feats.empty:
            raise HTTPException(422, "Not enough history to compute features")

        latest = feats.iloc[-1:]
        predicted, model_name = _predict_next(feats)
        last_close = float(latest["Close"].iloc[0])
        as_of = pd.Timestamp(latest["Date"].iloc[0]).date()

        change = predicted - last_close
        pct = change / last_close * 100.0

        # Empirical 80% interval from the recent realised volatility of the
        # series, not from the model — none of the point forecasters here
        # produce a calibrated predictive distribution.
        sigma = float(feats["log_ret"].iloc[:-1].tail(60).std())
        lo, hi = predicted * (1 - 1.2816 * sigma), predicted * (1 + 1.2816 * sigma)

        COUNTERS["latency_sum_ms"] += (time.perf_counter() - started) * 1000
        MONITOR.record(as_of=as_of, model=model_name,
                       last_close=last_close, predicted_close=predicted,
                       change_pct=pct)
        return ForecastResponse(
            ticker=DATA.ticker,
            model=model_name,
            as_of=as_of,
            last_close=round(last_close, 4),
            predicted_close=round(predicted, 4),
            predicted_change=round(change, 4),
            predicted_change_pct=round(pct, 4),
            direction="up" if change > 0 else ("down" if change < 0 else "flat"),
            horizon_days=DATA.horizon,
            prediction_interval_80=(round(lo, 4), round(hi, 4)),
            disclaimer=DISCLAIMER,
        )

    except HTTPException:
        COUNTERS["forecast_errors"] += 1
        raise
    except Exception as err:
        COUNTERS["forecast_errors"] += 1
        log.exception("Forecast failed")
        raise HTTPException(500, f"Forecast failed: {err}") from err


@app.get("/predictions", tags=["ops"])
def prediction_distribution() -> dict:
    """Shape of recent forecasts.

    Latency and error rate say the service is up; this says whether it has
    started emitting nonsense. `implausible_rate_pct` is the alarm signal.
    """
    return MONITOR.distribution()


@app.get("/drift", tags=["ops"])
def drift() -> dict:
    """Data drift against the training distribution (PSI + KS)."""
    try:
        return drift_summary()
    except Exception as err:
        log.exception("Drift check failed")
        raise HTTPException(500, f"Drift check failed: {err}") from err


@app.get("/", tags=["ops"])
def root() -> dict:
    return {
        "service": "NVDA Stock Price Forecasting API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/model", "/forecast", "/metrics",
                      "/predictions", "/drift"],
        "disclaimer": DISCLAIMER,
    }
