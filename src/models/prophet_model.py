"""
src/models/prophet_model.py
────────────────────────────
Meta / Facebook Prophet wrapper for NVDA stock price forecasting.

Features:
  • Automatic holiday detection (US market holidays)
  • Optional regressors (volume, technical indicators)
  • Prophet cross-validation + performance metrics
  • Confidence-interval forecast output
  • JSON serialisation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    FORECAST_DAYS,
    MODELS_DIR,
    PROPHET_CHANGEPOINT_PRIOR,
    PROPHET_CV_HORIZON,
    PROPHET_CV_INITIAL,
    PROPHET_CV_PERIOD,
    PROPHET_HOLIDAYS_PRIOR,
    PROPHET_SEASONALITY_PRIOR,
    PROPHET_WEEKLY_SEASONALITY,
    PROPHET_YEARLY_SEASONALITY,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Wraps Facebook Prophet for stock price forecasting.

    Usage:
        pf = ProphetForecaster()
        pf.fit(train_df)                         # train_df must have Date index + Close
        forecast = pf.predict(periods=30)
        pf.save(path)
    """

    def __init__(
        self,
        changepoint_prior_scale:  float = PROPHET_CHANGEPOINT_PRIOR,
        seasonality_prior_scale:  float = PROPHET_SEASONALITY_PRIOR,
        holidays_prior_scale:     float = PROPHET_HOLIDAYS_PRIOR,
        yearly_seasonality:       bool  = PROPHET_YEARLY_SEASONALITY,
        weekly_seasonality:       bool  = PROPHET_WEEKLY_SEASONALITY,
        regressors:               Optional[List[str]] = None,
        log_transform:            bool  = True,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale    = holidays_prior_scale
        self.yearly_seasonality      = yearly_seasonality
        self.weekly_seasonality      = weekly_seasonality
        self.regressors              = regressors or []
        self._model                  = None
        self._last_ds_               = None
        # Fit on log(price): NVIDIA grew ~5,000x, so an additive model on raw
        # prices is dominated by the last few years. Log makes growth linear.
        self.log_transform           = log_transform

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_prophet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert OHLCV DataFrame to Prophet's (ds, y) format."""
        pdf = pd.DataFrame({
            "ds": df.index.tz_localize(None) if df.index.tz else df.index,
            "y":  (np.log(df[TARGET_COL].astype(float).values)
                   if self.log_transform else df[TARGET_COL].values),
        })
        for col in self.regressors:
            if col in df.columns:
                pdf[col] = df[col].values
        return pdf.dropna(subset=["ds", "y"])

    def _get_us_holidays(self) -> pd.DataFrame:
        """Build US stock-market holidays DataFrame for Prophet."""
        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            cal   = USFederalHolidayCalendar()
            rules = cal.rules
            # Generate holidays for 1999-2030
            holidays = cal.holidays(start="1999-01-01", end="2030-12-31")
            return pd.DataFrame({
                "holiday": "US_market_holiday",
                "ds":      pd.to_datetime(holidays),
            })
        except Exception:
            logger.warning("Could not build US holiday calendar; skipping.")
            return pd.DataFrame(columns=["holiday", "ds"])

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        """
        Fit Prophet on the training DataFrame.
        df must have a DatetimeIndex and a 'Close' column.
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "prophet is not installed. Run: pip install prophet"
            )

        pdf = self._to_prophet_df(df)
        holidays = self._get_us_holidays()

        self._model = Prophet(
            changepoint_prior_scale  = self.changepoint_prior_scale,
            seasonality_prior_scale  = self.seasonality_prior_scale,
            holidays_prior_scale     = self.holidays_prior_scale,
            yearly_seasonality       = self.yearly_seasonality,
            weekly_seasonality       = self.weekly_seasonality,
            daily_seasonality        = False,
            holidays                 = holidays if not holidays.empty else None,
        )

        # Add extra regressors
        for reg in self.regressors:
            if reg in pdf.columns:
                self._model.add_regressor(reg)

        self._model.fit(pdf)
        self._last_ds_ = pdf["ds"].max()
        logger.info("Prophet fitted on %d rows, last date: %s",
                    len(pdf), self._last_ds_)
        return self

    # ── Forecasting ───────────────────────────────────────────────────────────

    def predict(
        self,
        periods: int = FORECAST_DAYS,
        freq: str = "B",         # business days
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce a forecast DataFrame from the model's last training date.

        Returns a DataFrame with columns:
            ds, yhat, yhat_lower, yhat_upper, trend, weekly, yearly
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        future = self._model.make_future_dataframe(
            periods=periods, freq=freq, include_history=False
        )

        if self.regressors and future_regressors is not None:
            for reg in self.regressors:
                if reg in future_regressors.columns:
                    future[reg] = future_regressors[reg].values[: len(future)]

        forecast = self._model.predict(future)
        logger.info("Prophet forecast generated: %d steps", len(forecast))
        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper",
                        "trend", "additive_terms"]].copy()
        if self.log_transform:
            for col in ["yhat", "yhat_lower", "yhat_upper"]:
                out[col] = np.exp(out[col])
        return out

    def _raw_yhat(self, dates) -> pd.DataFrame:
        """Model-space (log) yhat and interval for the given dates."""
        future = pd.DataFrame({"ds": pd.to_datetime(list(dates))})
        return self._model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def one_step_predictions(self, prices: pd.Series) -> np.ndarray:
        """
        One-step-ahead predictions for prices[1:]:
            pred_t = price_{t-1} * exp(yhat_t - yhat_{t-1})
        i.e. Prophet supplies the expected drift + seasonality for the next
        day, anchored on the last observed close (no look-ahead).
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predicting.")
        yh = self._raw_yhat(prices.index)["yhat"].values
        if not self.log_transform:
            return prices.values[:-1] + np.diff(yh)
        return prices.values[:-1] * np.exp(np.diff(yh))

    def forecast_path(self, last_date, last_close: float, steps: int = FORECAST_DAYS):
        """
        Forward price path for the next `steps` business days, anchored on
        the latest close. Returns (dates, prices, lower, upper).
        """
        dates = pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=steps)
        fc = self._raw_yhat([pd.Timestamp(last_date)] + list(dates))
        base = fc["yhat"].iloc[0]
        rel = fc[["yhat", "yhat_lower", "yhat_upper"]].iloc[1:].values - base
        if self.log_transform:
            vals = last_close * np.exp(rel)
        else:
            vals = last_close + rel
        return dates, vals[:, 0], vals[:, 1], vals[:, 2]

    def predict_on_history(self) -> pd.DataFrame:
        """Get in-sample predictions (useful for residual analysis)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_on_history().")
        future = self._model.make_future_dataframe(periods=0, freq="B",
                                                    include_history=True)
        return self._model.predict(future)

    # ── Cross-validation ──────────────────────────────────────────────────────

    def cross_validate(
        self,
        initial: str = PROPHET_CV_INITIAL,
        period:  str = PROPHET_CV_PERIOD,
        horizon: str = PROPHET_CV_HORIZON,
    ) -> Dict[str, float]:
        """
        Run Prophet's built-in cross-validation and return metrics.
        """
        if self._model is None:
            raise RuntimeError("Fit the model first.")

        try:
            from prophet.diagnostics import cross_validation, performance_metrics

            df_cv = cross_validation(
                self._model, initial=initial, period=period, horizon=horizon,
                parallel="processes",
            )
            df_pm = performance_metrics(df_cv)
            metrics = {
                "MAE":  float(df_pm["mae"].mean()),
                "RMSE": float(df_pm["rmse"].mean()),
                "MAPE": float(df_pm["mape"].mean()) * 100,
            }
            logger.info("Prophet CV -> %s", metrics)
            return metrics
        except Exception as exc:
            logger.warning("Prophet CV failed: %s", exc)
            return {}

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Serialise using Prophet's JSON serialiser."""
        path = Path(path or MODELS_DIR / "prophet_model.json")
        if self._model is None:
            raise RuntimeError("Nothing to save — model not fitted.")
        try:
            from prophet.serialize import model_to_json

            with open(path, "w") as f:
                json.dump(model_to_json(self._model), f)
            logger.info("Prophet model saved -> %s", path)
        except ImportError:
            import pickle
            pkl_path = path.with_suffix(".pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self._model, f)
            logger.info("Prophet model saved (pickle) -> %s", pkl_path)
            path = pkl_path
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ProphetForecaster":
        path = Path(path or MODELS_DIR / "prophet_model.json")
        obj  = cls()

        if path.suffix == ".json":
            try:
                from prophet.serialize import model_from_json

                with open(path) as f:
                    obj._model = model_from_json(json.load(f))
                logger.info("Prophet model loaded ← %s", path)
                return obj
            except ImportError:
                pass

        pkl_path = path.with_suffix(".pkl")
        if pkl_path.exists():
            import pickle
            with open(pkl_path, "rb") as f:
                obj._model = pickle.load(f)
            logger.info("Prophet model loaded (pickle) ← %s", pkl_path)

        return obj
