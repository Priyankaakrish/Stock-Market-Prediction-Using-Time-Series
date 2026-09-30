"""
api/routers/predict.py
───────────────────────
FastAPI router for all prediction endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    SinglePredictRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predictions"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _registry(request: Request) -> dict:
    return getattr(request.app.state, "model_registry", {}) or {}


def _run_forecast(registry: dict, steps: int, which: List[str]) -> dict:
    """Forward forecasts {model: array} using the shared forecasting logic."""
    from src.forecasting import forecast_all

    ohlcv = registry.get("ohlcv")
    if ohlcv is None:
        raise HTTPException(503, "Price history not loaded.")
    return forecast_all(
        registry.get("models", {}), ohlcv, steps=steps,
        scaler=registry.get("scaler"), feat_cols=registry.get("feature_cols"),
        which=which,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Returns API health status and loaded model names."""
    registry = _registry(request)
    return HealthResponse(
        status="healthy",
        models_loaded=list(registry.get("models", {}).keys()),
        last_data_date=str(registry.get("last_data_date", "unknown")),
    )


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(body: ForecastRequest, request: Request):
    """
    Multi-step forecast endpoint.

    - Accepts a list of model names and a step count.
    - Returns per-model forecasts + ensemble blend.
    - Confidence intervals included when requested.
    """
    registry  = _registry(request)
    models    = registry.get("models", {})
    last_date = registry.get("last_date_series")
    steps     = body.steps

    if not models:
        raise HTTPException(503, "No models loaded. Run train_pipeline.py first.")

    # "ensemble" needs every base model
    wanted = [m for m in body.models if m != "ensemble"]
    if "ensemble" in body.models:
        wanted = list(models.keys())
    preds_all = _run_forecast(registry, steps, wanted)

    requested = set(body.models)
    preds_dict = {k: v for k, v in preds_all.items()
                  if k in requested or "ensemble" not in requested}

    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    ensemble_preds = ens_lower = ens_upper = None
    if "ensemble" in requested and preds_all:
        ens_model = registry.get("ensemble")
        if ens_model:
            ens_df = ens_model.predict(preds_all, compute_ci=True)
            ensemble_preds = ens_df["ensemble"].values
            if body.include_confidence_interval and "lower" in ens_df.columns:
                ens_lower = ens_df["lower"].values
                ens_upper = ens_df["upper"].values

    forecasts_out = {
        name: [ForecastPoint(date=str(future_dates[i].date()),
                             predicted=round(float(arr[i]), 4))
               for i in range(min(steps, len(arr)))]
        for name, arr in preds_dict.items()
    }

    ensemble_out = None
    if ensemble_preds is not None:
        ensemble_out = [
            ForecastPoint(
                date=str(future_dates[i].date()),
                predicted=round(float(ensemble_preds[i]), 4),
                lower_bound=round(float(ens_lower[i]), 4) if ens_lower is not None else None,
                upper_bound=round(float(ens_upper[i]), 4) if ens_upper is not None else None,
            )
            for i in range(min(steps, len(ensemble_preds)))
        ]

    return ForecastResponse(
        ticker="NVDA",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        forecast_days=steps,
        models_used=list(preds_dict.keys()) + (["ensemble"] if ensemble_out else []),
        forecasts=forecasts_out,
        ensemble=ensemble_out,
        metadata={
            "last_known_date": str(last_date.date()) if last_date is not None else None,
            "ci_included":     body.include_confidence_interval,
        },
    )


@router.get("/models", response_model=List[ModelInfoResponse])
async def list_models(request: Request):
    """List all loaded models and their metadata."""
    models = _registry(request).get("models", {})
    infos  = []
    for name, model in models.items():
        info = ModelInfoResponse(model=name)
        params = None
        if isinstance(getattr(model, "best_params_", None), dict):
            params = model.best_params_
        elif getattr(model, "order", None) is not None:
            params = {"order": str(model.order),
                      "aic": float(getattr(model, "best_aic_", None) or 0)}
        info.parameters = params
        infos.append(info)
    return infos


@router.post("/predict/single", response_model=ForecastPoint)
async def predict_single(body: SinglePredictRequest, request: Request):
    """
    Predicted close `steps` business days ahead of the latest data, using the
    XGBoost model (fastest inference).
    """
    registry = _registry(request)
    if "xgboost" not in registry.get("models", {}):
        raise HTTPException(503, "XGBoost model not loaded.")

    preds = _run_forecast(registry, body.steps, ["xgboost"]).get("xgboost")
    if preds is None or not len(preds):
        raise HTTPException(500, "XGBoost forecast failed.")

    last_date    = registry.get("last_date_series", pd.Timestamp.now())
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=body.steps)
    return ForecastPoint(date=str(future_dates[-1].date()),
                         predicted=round(float(preds[-1]), 4))
