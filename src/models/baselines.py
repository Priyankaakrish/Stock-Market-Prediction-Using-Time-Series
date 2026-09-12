"""Baselines. These exist to keep the sophisticated models honest.

The random-walk ("naive") forecast — tomorrow's close equals today's close —
is the theoretical optimum under the weak-form efficient market hypothesis.
Published stock-prediction results that report a 0.5% MAPE and never show this
baseline are almost always reporting exactly this: a lagged copy of the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class NaiveForecaster(Forecaster):
    """ŷ(t+1) = y(t). The random walk."""

    name = "naive"

    def fit(self, train, val=None):
        return self

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return test["Close"].to_numpy(dtype=float)


class MovingAverageForecaster(Forecaster):
    """ŷ(t+1) = mean of the last ``window`` closes."""

    def __init__(self, window: int = 5):
        self.window = window
        self.name = f"ma{window}"
        self.min_history = window

    def fit(self, train, val=None):
        return self

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        series = pd.concat([history["Close"], test["Close"]], ignore_index=True)
        rolled = series.rolling(self.window).mean().to_numpy(dtype=float)
        return rolled[len(history):]

    def params(self) -> dict:
        return {"window": self.window}


class DriftForecaster(Forecaster):
    """Random walk with drift estimated from the training history."""

    name = "drift"

    def __init__(self):
        self.drift_ = 0.0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        closes = train["Close"].to_numpy(dtype=float)
        self.drift_ = float(
            (np.log(closes[-1]) - np.log(closes[0])) / max(len(closes) - 1, 1)
        )
        return self

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return test["Close"].to_numpy(dtype=float) * np.exp(self.drift_)

    def params(self) -> dict:
        return {"drift_log_per_day": round(self.drift_, 8)}
