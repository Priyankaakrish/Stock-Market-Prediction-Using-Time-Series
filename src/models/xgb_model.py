"""Gradient-boosted trees on engineered technical features.

The single most important decision in this file is the **target**: the model
learns ``log(Close_{t+1} / Close_t)`` — the next-day log return — and the price
forecast is reconstructed as ``Close_t * exp(ŷ)``.

Why not predict the price directly? A decision tree predicts the mean of the
leaf it lands in, so its output is bounded by the target range it saw during
training. Our training window ends around $12 (split-adjusted) while the test
window reaches $189. A price-level XGBoost would therefore emit a flat line at
its training maximum for the whole test period, scoring an RMSE of ~$70 while
looking superficially "trained". Predicting returns removes the level entirely
and keeps the problem stationary.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import MODELS
from ..features import feature_columns
from .base import Forecaster, returns_to_prices

log = logging.getLogger(__name__)


class XGBForecaster(Forecaster):
    name = "xgboost"
    min_history = 60

    def __init__(self, **params):
        self.params_ = {**MODELS.xgb_params, **params}
        self.model_ = None
        self.features_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        import xgboost as xgb

        self.features_ = feature_columns(train)
        X_tr = train[self.features_].to_numpy(dtype=float)
        y_tr = train["target_return"].to_numpy(dtype=float)

        params = dict(self.params_)
        fit_kwargs: dict = {}
        if val is not None and len(val):
            params["early_stopping_rounds"] = 50
            fit_kwargs["eval_set"] = [
                (val[self.features_].to_numpy(dtype=float),
                 val["target_return"].to_numpy(dtype=float))
            ]
            fit_kwargs["verbose"] = False

        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(X_tr, y_tr, **fit_kwargs)
        self.best_iteration_ = getattr(self.model_, "best_iteration", None)

        log.info("XGBoost fitted on %d rows x %d features (best_iter=%s)",
                 len(train), len(self.features_), self.best_iteration_)
        return self

    def refit_on(self, train: pd.DataFrame, val: pd.DataFrame) -> XGBForecaster:
        """Refit on train+val at the tuned number of trees, for final deployment."""
        import xgboost as xgb

        n_trees = (self.best_iteration_ + 1) if self.best_iteration_ else self.params_["n_estimators"]
        full = pd.concat([train, val], ignore_index=True)
        params = {**self.params_, "n_estimators": int(n_trees)}
        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(full[self.features_].to_numpy(dtype=float),
                        full["target_return"].to_numpy(dtype=float), verbose=False)
        log.info("XGBoost refitted on train+val (%d rows, %d trees)", len(full), n_trees)
        return self

    def predict_returns(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features_].to_numpy(dtype=float)
        return self.model_.predict(X).astype(float)

    def backtest(self, history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        preds = self.predict_returns(test)
        return returns_to_prices(test["Close"].to_numpy(dtype=float), preds)

    def feature_importance(self, top: int = 20) -> pd.DataFrame:
        imp = self.model_.feature_importances_
        return (
            pd.DataFrame({"feature": self.features_, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )

    def params(self) -> dict:
        return {
            **{f"xgb_{k}": v for k, v in self.params_.items()},
            "target": "log_return",
            "n_features": len(self.features_),
            "best_iteration": self.best_iteration_,
        }
