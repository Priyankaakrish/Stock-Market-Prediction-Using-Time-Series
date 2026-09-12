"""ARIMA(p,d,q) on log prices.

Modelling ``log(Close)`` with ``d=1`` means the model is really an ARMA on log
returns, which is the stationary representation of the series. Fitting ARIMA
directly on raw prices of a stock that grew 5,000x produces heteroskedastic
residuals and unstable coefficients.

For the backtest we use ``statsmodels``' ``append(refit=False)``: after each
test day the realised close is appended to the state so the next forecast is a
true one-step-ahead prediction, but the coefficients stay fixed. That is both
statistically correct and ~1000x faster than refitting daily.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..config import MODELS
from .base import Forecaster

log = logging.getLogger(__name__)


class ARIMAForecaster(Forecaster):
    name = "arima"
    min_history = 60

    def __init__(self, order: tuple[int, int, int] | None = None):
        self.order = tuple(order or MODELS.arima_order)
        self.res_ = None
        self.aic_ = float("nan")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        from statsmodels.tsa.arima.model import ARIMA

        hist = train if val is None else pd.concat([train, val], ignore_index=True)
        y = np.log(hist["Close"].to_numpy(dtype=float))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # With d=1 statsmodels forbids a constant term (it is annihilated
            # by differencing); a linear trend is the equivalent, and gives the
            # drift term we want in the differenced (log-return) representation.
            trend = "t" if self.order[1] > 0 else "c"
            model = ARIMA(y, order=self.order, trend=trend,
                          enforce_stationarity=False, enforce_invertibility=False)
            self.res_ = model.fit(method_kwargs={"warn_convergence": False})
        self.aic_ = float(self.res_.aic)
        log.info("ARIMA%s fitted on %d obs, AIC=%.1f", self.order, len(y), self.aic_)
        return self

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        if self.res_ is None:
            raise RuntimeError("Call fit() first")

        res = self.res_
        preds = np.empty(len(test), dtype=float)
        test_log = np.log(test["Close"].to_numpy(dtype=float))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(test)):
                preds[i] = float(res.forecast(steps=1)[0])
                # Append the realised value so day i+1 is still one-step-ahead.
                res = res.append([test_log[i]], refit=False)

        return np.exp(preds)

    def params(self) -> dict:
        return {"order": str(self.order), "aic": round(self.aic_, 2), "target": "log_close"}


def grid_search_order(train: pd.DataFrame, val: pd.DataFrame,
                      p_range=(0, 1, 2), d=1, q_range=(0, 1, 2)) -> tuple[int, int, int]:
    """Pick (p,d,q) by AIC on the training fold. Cheap, and good enough here."""
    from statsmodels.tsa.arima.model import ARIMA

    y = np.log(train["Close"].to_numpy(dtype=float))
    best, best_aic = (1, d, 1), np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    res = ARIMA(y, order=(p, d, q), trend="t" if d > 0 else "c",
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit(
                        method_kwargs={"warn_convergence": False})
                    if res.aic < best_aic:
                        best, best_aic = (p, d, q), float(res.aic)
                except Exception:
                    continue
    log.info("Best ARIMA order by AIC: %s (AIC=%.1f)", best, best_aic)
    return best
