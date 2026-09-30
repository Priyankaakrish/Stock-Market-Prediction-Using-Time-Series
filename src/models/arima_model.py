"""
src/models/arima_model.py
─────────────────────────
ARIMA / SARIMA model wrapper with:
  • Auto-order selection via AIC grid search or pmdarima auto_arima
  • Walk-forward (expanding-window) cross-validation
  • Forecast with confidence intervals
  • Serialisation via pickle
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ARIMA_IC,
    ARIMA_MAX_D,
    ARIMA_MAX_P,
    ARIMA_MAX_Q,
    ARIMA_SUBSAMPLE,
    FORECAST_DAYS,
    MODELS_DIR,
    SARIMA_S,
    TARGET_COL,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """
    Thin wrapper around statsmodels ARIMA / SARIMAX.

    Usage:
        model = ARIMAForecaster()
        model.auto_fit(train_series)
        preds, ci = model.forecast(steps=30)
        model.save(path)
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        use_auto: bool = True,
    ):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.use_auto       = use_auto
        self._model         = None
        self._result        = None
        self.best_aic_      = None
        self.log_transform  = False   # True -> model works on log(price)
        self.trend          = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def auto_fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Use pmdarima.auto_arima for order selection (preferred),
        with fallback to manual grid search.

        Subsampling: ARIMA is a short-memory model — fitting on all 4 k+ rows
        gives no accuracy benefit but costs 20-40 min on each run.  We use only
        the most recent ARIMA_SUBSAMPLE rows (default 500) instead.
        """
        series = series.dropna()

        # ── Subsample to the most recent N observations ────────────────────────
        if ARIMA_SUBSAMPLE and len(series) > ARIMA_SUBSAMPLE:
            logger.info(
                "ARIMA subsample: using last %d of %d rows (full series not "
                "needed — ARIMA is short-memory).",
                ARIMA_SUBSAMPLE, len(series),
            )
            series = series.iloc[-ARIMA_SUBSAMPLE:]

        # ── Seasonal flag: m=0 means "no seasonal search" ─────────────────────
        use_seasonal = (SARIMA_S is not None and SARIMA_S > 1)

        try:
            import pmdarima as pm
            logger.info(
                "Running auto_arima (pmdarima) — max_p=%d max_q=%d seasonal=%s …",
                ARIMA_MAX_P, ARIMA_MAX_Q, use_seasonal,
            )
            kwargs = dict(
                start_p=0, max_p=ARIMA_MAX_P,
                start_q=0, max_q=ARIMA_MAX_Q,
                d=None,
                max_d=ARIMA_MAX_D,
                information_criterion=ARIMA_IC,
                stepwise=True,        # Hyndman-Khandakar stepwise — much faster
                error_action="ignore",
                suppress_warnings=True,
                n_jobs=1,             # parallel fitting sometimes hangs on Windows
            )
            if use_seasonal:
                kwargs.update(seasonal=True, m=SARIMA_S,
                              max_P=2, max_D=1, max_Q=2, n_fits=30)
            else:
                kwargs.update(seasonal=False)

            result = pm.auto_arima(series, **kwargs)
            self.order          = result.order
            self.seasonal_order = result.seasonal_order
            self.best_aic_      = result.aic()
            self._result        = result
            logger.info("auto_arima -> order=%s seasonal=%s AIC=%.2f",
                        self.order, self.seasonal_order, self.best_aic_)

        except ImportError:
            logger.warning("pmdarima not installed — falling back to grid search")
            self._grid_search(series)

        return self

    def _grid_search(self, series: pd.Series) -> None:
        """Manual AIC grid search over p, d, q space."""
        from itertools import product
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best_aic   = np.inf
        best_order = (1, 1, 1)

        for p, d, q in product(
            range(ARIMA_MAX_P + 1),
            range(ARIMA_MAX_D + 1),
            range(ARIMA_MAX_Q + 1),
        ):
            try:
                res = SARIMAX(series, order=(p, d, q)).fit(disp=False)
                if res.aic < best_aic:
                    best_aic   = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue

        self.order          = best_order
        self.seasonal_order = (0, 0, 0, 0)
        self.best_aic_      = best_aic
        logger.info("Grid search -> order=%s AIC=%.2f", self.order, best_aic)
        self._fit_statsmodels(series)

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """Fit with a pre-specified order (skips auto selection)."""
        assert self.order is not None, "Set self.order before calling fit()"
        self._fit_statsmodels(series)
        return self

    def _fit_statsmodels(self, series: pd.Series) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        s_order = self.seasonal_order or (0, 0, 0, 0)
        self._result = SARIMAX(
            series,
            order=self.order,
            seasonal_order=s_order,
        ).fit(disp=False)
        self.best_aic_ = self._result.aic

    # ── Log-price workflow (used by the NVDA pipeline) ───────────────────────

    def _build_sm(self, y: np.ndarray):
        from statsmodels.tsa.arima.model import ARIMA
        return ARIMA(y, order=self.order,
                     seasonal_order=self.seasonal_order or (0, 0, 0, 0),
                     trend=self.trend)

    def fit_log(self, prices: pd.Series, reuse_order: bool = False) -> "ARIMAForecaster":
        """
        Fit on log(price). Modelling log prices turns NVIDIA's exponential
        growth into a roughly linear drift and makes errors proportional.
        Order is chosen by auto_arima unless `reuse_order` is True.
        """
        y = np.log(prices.dropna().astype(float))
        if ARIMA_SUBSAMPLE and len(y) > ARIMA_SUBSAMPLE:
            y = y.iloc[-ARIMA_SUBSAMPLE:]
        if not (reuse_order and self.order):
            self.auto_fit(y.reset_index(drop=True))
        d = self.order[1]
        self.trend = "t" if d == 1 else ("n" if d > 1 else "c")
        self._result = self._build_sm(y.values).fit()
        self.best_aic_ = float(self._result.aic)
        self.log_transform = True
        logger.info("ARIMA%s on log-price fitted (trend=%s, AIC=%.2f)",
                    self.order, self.trend, self.best_aic_)
        return self

    def one_step_predictions(self, prices: pd.Series, start: int) -> np.ndarray:
        """
        One-step-ahead price predictions for prices[start:], each made with
        data up to the previous day only. Uses the fitted parameters (no refit).
        """
        y = np.log(prices.astype(float).values)
        res = self._build_sm(y).filter(self._result.params)
        pred = res.get_prediction(start=start).predicted_mean
        return np.exp(np.asarray(pred))

    # ── Forecasting ───────────────────────────────────────────────────────────

    def forecast(
        self,
        steps: int = FORECAST_DAYS,
        alpha: float = 0.05,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Multi-step forecast.

        Returns:
            predictions : pd.Series  (point forecasts)
            conf_int    : pd.DataFrame  (lower / upper columns)
        """
        if self._result is None:
            raise RuntimeError("Model not fitted yet.")

        if not hasattr(self._result, "get_forecast"):
            # pmdarima path
            preds, ci = self._result.predict(
                n_periods=steps, return_conf_int=True, alpha=alpha
            )
            preds = pd.Series(np.asarray(preds))
            ci    = pd.DataFrame(np.asarray(ci), columns=["lower", "upper"])
        else:
            # statsmodels path
            fc = self._result.get_forecast(steps=steps)
            preds = pd.Series(np.asarray(fc.predicted_mean))
            ci    = pd.DataFrame(np.asarray(fc.conf_int(alpha=alpha)),
                                 columns=["lower", "upper"])

        if self.log_transform:
            preds = np.exp(preds)
            ci    = np.exp(ci)
        return preds, ci

    def update_and_predict(
        self,
        new_obs: float,
        steps: int = 1,
    ) -> float:
        """Append a new observation and predict the next step (online update)."""
        if hasattr(self._result, "update"):
            self._result = self._result.append([new_obs], refit=False)
        preds, _ = self.forecast(steps=steps)
        return float(preds.iloc[0])

    # ── Walk-forward CV ───────────────────────────────────────────────────────

    def walk_forward_cv(
        self,
        series: pd.Series,
        n_splits: int = 5,
        horizon: int = 30,
    ) -> Dict[str, float]:
        """
        Expanding-window walk-forward validation.
        Reports MAE, RMSE, MAPE, Directional Accuracy.
        """
        from src.evaluation.metrics import regression_metrics

        n = len(series)
        initial_train = int(n * 0.6)
        step = (n - initial_train - horizon) // n_splits

        all_preds, all_actuals = [], []

        for i in range(n_splits):
            train_end = initial_train + i * step
            train_s   = series.iloc[:train_end]
            actual_s  = series.iloc[train_end: train_end + horizon]

            if len(actual_s) == 0:
                break

            try:
                clone = ARIMAForecaster(
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    use_auto=False,
                )
                clone._fit_statsmodels(train_s)
                preds, _ = clone.forecast(steps=len(actual_s))

                all_preds.extend(preds.values[:len(actual_s)])
                all_actuals.extend(actual_s.values)
            except Exception as exc:
                logger.warning("CV fold %d failed: %s", i, exc)

        if not all_preds:
            return {}

        return regression_metrics(
            np.array(all_actuals), np.array(all_preds), label="ARIMA_CV"
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or MODELS_DIR / "arima_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({"order": self.order,
                         "seasonal_order": self.seasonal_order,
                         "result": self._result,
                         "aic": self.best_aic_,
                         "log_transform": self.log_transform,
                         "trend": self.trend}, f)
        logger.info("ARIMA model saved -> %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ARIMAForecaster":
        path = Path(path or MODELS_DIR / "arima_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(order=data["order"],
                    seasonal_order=data["seasonal_order"],
                    use_auto=False)
        model._result   = data["result"]
        model.best_aic_ = data["aic"]
        model.log_transform = data.get("log_transform", False)
        model.trend         = data.get("trend")
        logger.info("ARIMA model loaded ← %s", path)
        return model
