from .base import Forecaster
from .baselines import DriftForecaster, MovingAverageForecaster, NaiveForecaster

__all__ = [
    "ARIMAForecaster",
    "DriftForecaster",
    "Forecaster",
    "LSTMForecaster",
    "MovingAverageForecaster",
    "NaiveForecaster",
    "ProphetForecaster",
    "XGBForecaster",
]


def __getattr__(name: str):
    """Lazy imports so a missing optional dependency (prophet, torch) does not
    break the whole package at import time."""
    if name == "ARIMAForecaster":
        from .arima_model import ARIMAForecaster
        return ARIMAForecaster
    if name == "ProphetForecaster":
        from .prophet_model import ProphetForecaster
        return ProphetForecaster
    if name == "XGBForecaster":
        from .xgb_model import XGBForecaster
        return XGBForecaster
    if name == "LSTMForecaster":
        from .lstm_model import LSTMForecaster
        return LSTMForecaster
    raise AttributeError(name)
