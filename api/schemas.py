"""Request/response contracts for the forecasting service."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class OHLCVBar(BaseModel):
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("high")
    @classmethod
    def _high_ge_low(cls, v, info):
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError("high must be >= low")
        return v


class ForecastRequest(BaseModel):
    """Optionally supply your own history; otherwise the server uses the
    most recent bars from its own cached dataset."""

    bars: list[OHLCVBar] | None = Field(
        default=None,
        description="Chronologically ordered daily bars. Needs at least "
                    "~120 rows for the technical indicators to warm up.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"bars": None}]
        }
    }


class ForecastResponse(BaseModel):
    ticker: str
    model: str
    as_of: date = Field(description="Last observed trading day")
    last_close: float
    predicted_close: float
    predicted_change: float
    predicted_change_pct: float
    direction: str
    horizon_days: int
    prediction_interval_80: tuple[float, float] | None = None
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    trained_at: str | None
    data_rows: int
    data_last_date: str | None
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str | None
    horizon_days: int
    train_end: str | None
    test_window: str | None
    test_metrics: dict
    naive_rmse: float | None
    mlflow_run_id: str | None
