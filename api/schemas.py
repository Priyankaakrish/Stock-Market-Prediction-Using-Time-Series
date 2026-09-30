"""
api/schemas.py
───────────────
Pydantic v2 request / response schemas for the prediction API.
"""

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """Single-model or ensemble forecast request."""
    steps: int = Field(
        default=30,
        ge=1,
        le=252,
        description="Number of business days to forecast (1–252)",
    )
    models: List[str] = Field(
        default=["ensemble"],
        description="Which model(s) to use: arima|prophet|lstm|xgboost|ensemble",
    )
    include_confidence_interval: bool = Field(
        default=True,
        description="Include 95 % confidence interval in the response",
    )

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        allowed = {"arima", "prophet", "lstm", "xgboost", "ensemble"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Unknown model(s): {invalid}. Choose from {allowed}.")
        return v


class SinglePredictRequest(BaseModel):
    """Predict a single future date given current OHLCV context."""
    close:  float = Field(..., description="Latest closing price", gt=0)
    open:   Optional[float] = Field(None, description="Latest open price", gt=0)
    high:   Optional[float] = Field(None, description="Latest high price", gt=0)
    low:    Optional[float] = Field(None, description="Latest low price", gt=0)
    volume: Optional[float] = Field(None, description="Latest volume", ge=0)
    steps:  int             = Field(1, ge=1, le=10,
                                     description="Steps ahead (1–10)")


# ── Response schemas ──────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    date:        str
    predicted:   float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ModelForecast(BaseModel):
    model:      str
    forecasts:  List[ForecastPoint]
    mae:        Optional[float] = None
    rmse:       Optional[float] = None


class ForecastResponse(BaseModel):
    ticker:          str = "NVDA"
    generated_at:    str
    forecast_days:   int
    models_used:     List[str]
    forecasts:       Dict[str, List[ForecastPoint]]
    ensemble:        Optional[List[ForecastPoint]] = None
    metadata:        Optional[Dict] = None


class HealthResponse(BaseModel):
    status:          str
    models_loaded:   List[str]
    last_data_date:  Optional[str] = None
    version:         str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model:       str
    trained_at:  Optional[str] = None
    parameters:  Optional[Dict] = None
    metrics:     Optional[Dict] = None
