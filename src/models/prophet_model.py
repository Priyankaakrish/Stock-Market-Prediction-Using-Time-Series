"""Facebook/Meta Prophet on log prices.

Prophet decomposes a series into trend + seasonality + holidays. That is a good
fit for demand data with genuine calendar structure; it is a poor structural
match for an equity price, which is close to a random walk with no meaningful
weekly or yearly seasonality. We include it anyway because the brief calls for
it, and because the result is informative: it shows what happens when you
impose a smooth trend model on a martingale.

Backtest protocol: retrain every ``refit_every`` trading days on all data seen
so far, then roll the forecast forward until the next retrain. Daily refits
would be more accurate but cost ~1,000 Stan fits per evaluation, which is not
what anyone runs in production.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import MODELS
from .base import Forecaster

log = logging.getLogger(__name__)


def _silence_stan() -> None:
    import logging as _logging

    for noisy in ("prophet", "cmdstanpy", "prophet.models", "prophet.forecaster"):
        _logging.getLogger(noisy).setLevel(_logging.CRITICAL)


class ProphetForecaster(Forecaster):
    name = "prophet"
    min_history = 365

    def __init__(self, refit_every: int | None = None, **kwargs):
        self.refit_every = refit_every or MODELS.prophet_refit_every
        self.kwargs = {**MODELS.prophet_params, **kwargs}
        self.model_ = None
        self._train_tail: pd.DataFrame | None = None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _frame(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ds": pd.to_datetime(df["Date"]),
                "y": np.log(df["Close"].to_numpy(dtype=float)),
            }
        )

    def _fit_one(self, df: pd.DataFrame):
        from prophet import Prophet

        _silence_stan()
        m = Prophet(**self.kwargs)
        m.fit(self._frame(df))
        return m

    # -- interface ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        hist = train if val is None else pd.concat([train, val], ignore_index=True)
        self.model_ = self._fit_one(hist)
        self._train_tail = hist.tail(400)[["Date", "Close"]].copy()
        log.info("Prophet fitted on %d observations", len(hist))
        return self

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        hist = history[["Date", "Close"]].copy()
        preds = np.empty(len(test), dtype=float)
        model = self.model_ if self.model_ is not None else self._fit_one(hist)

        i = 0
        while i < len(test):
            block = test.iloc[i: i + self.refit_every]
            future = pd.DataFrame({"ds": pd.to_datetime(block["Date"])})
            fc = model.predict(future)["yhat"].to_numpy(dtype=float)
            preds[i: i + len(block)] = np.exp(fc)

            # Absorb the realised block, then retrain for the next chunk.
            hist = pd.concat([hist, block[["Date", "Close"]]], ignore_index=True)
            i += len(block)
            if i < len(test):
                model = self._fit_one(hist)
                log.debug("Prophet retrained at test index %d", i)

        return preds

    def params(self) -> dict:
        return {"refit_every_days": self.refit_every, "target": "log_close", **self.kwargs}
