"""Common interface for every forecaster in the project.

The contract is deliberately narrow so that ARIMA, Prophet, XGBoost, an LSTM
and a two-line naive rule can all be scored by exactly the same harness.

``backtest`` is the honest evaluation entry point: it produces a one-step-ahead
prediction for every row of the test set, using only information available at
that point in time. Models that need to see realised values as they arrive
(ARIMA) or that retrain periodically (Prophet) implement that inside
``backtest`` rather than cheating by fitting on the test window.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = "base"
    #: Set by subclasses that require a lookback buffer at serving time.
    min_history: int = 1

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> Forecaster:
        """Fit on data strictly before the evaluation window."""

    @abstractmethod
    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """One-step-ahead predictions of ``test['target']`` (next-day close)."""

    def predict_next(self, recent: pd.DataFrame) -> float:
        """Single next-day close forecast given the most recent feature rows."""
        return float(self.backtest(recent.iloc[:-1], recent.iloc[-1:])[0])

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> Forecaster:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def params(self) -> dict:
        return {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r}>"


def returns_to_prices(last_close: np.ndarray, log_returns: np.ndarray) -> np.ndarray:
    """Convert predicted next-day log returns back into price levels."""
    lr = np.clip(np.asarray(log_returns, dtype=float), -0.5, 0.5)
    return np.asarray(last_close, dtype=float) * np.exp(lr)
