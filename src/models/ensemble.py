"""
src/models/ensemble.py
───────────────────────
Weighted ensemble that blends ARIMA, Prophet, LSTM, and XGBoost forecasts.

Two blending strategies:
  1. Fixed weights    — from config.ENSEMBLE_WEIGHTS (default)
  2. Optimised weights— scipy minimize on validation-set RMSE

Output format: pd.DataFrame with columns
  [date, arima, prophet, lstm, xgboost, ensemble, lower, upper]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import ENSEMBLE_WEIGHTS, FORECAST_DAYS, MODELS_DIR

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Combines individual model predictions into a single forecast.

    Usage:
        ens = EnsembleForecaster()
        ens.fit_weights(val_preds_dict, val_actuals)   # optimise
        forecast = ens.predict(preds_dict)
        ens.save()
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights     = weights or dict(ENSEMBLE_WEIGHTS)
        self.optimised_  = False
        self.model_names = list(self.weights.keys())

    # ── Weight optimisation ───────────────────────────────────────────────────

    def fit_weights(
        self,
        preds_dict: Dict[str, np.ndarray],   # {model_name: preds_array}
        actuals:    np.ndarray,
    ) -> Dict[str, float]:
        """
        Optimise ensemble weights to minimise RMSE on validation predictions.
        Uses scipy L-BFGS-B with simplex constraints (weights sum to 1, all ≥ 0).
        """
        names  = [n for n in self.model_names if n in preds_dict]
        P      = np.column_stack([preds_dict[n] for n in names])  # (T, n_models)
        n_mdls = P.shape[1]

        def objective(w):
            ensemble_pred = P @ w
            return float(np.sqrt(np.mean((ensemble_pred - actuals) ** 2)))

        # Constraints: weights ≥ 0, sum = 1
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds      = [(0, 1)] * n_mdls
        w0          = np.array([1 / n_mdls] * n_mdls)

        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if result.success:
            self.weights = {n: float(w) for n, w in zip(names, result.x)}
            self.optimised_ = True
            logger.info("Ensemble weights optimised -> %s (RMSE=%.4f)",
                        self.weights, result.fun)
        else:
            logger.warning("Weight optimisation did not converge; using defaults.")

        return self.weights

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        preds_dict: Dict[str, np.ndarray],
        compute_ci: bool = True,
        ci_std_multiplier: float = 1.96,
    ) -> pd.DataFrame:
        """
        Blend individual predictions into an ensemble forecast.

        Args:
            preds_dict : {model_name: array of shape (T,)}
            compute_ci : whether to estimate confidence intervals
                         via prediction spread across models

        Returns:
            pd.DataFrame with columns per-model + 'ensemble' + 'lower' + 'upper'
        """
        result = {}
        T = None

        for name in self.model_names:
            if name not in preds_dict:
                logger.warning("Model '%s' missing from preds_dict; skipping.", name)
                continue
            arr = np.array(preds_dict[name]).flatten()
            result[name] = arr
            if T is None:
                T = len(arr)

        if not result:
            raise ValueError("No valid predictions provided to ensemble.")

        # Weighted blend
        names   = list(result.keys())
        weights = np.array([self.weights.get(n, 0.0) for n in names])
        weights /= weights.sum()   # re-normalise in case of absent models

        P         = np.column_stack([result[n] for n in names])
        ensemble  = P @ weights
        result["ensemble"] = ensemble

        if compute_ci:
            pred_std = P.std(axis=1)
            result["lower"] = ensemble - ci_std_multiplier * pred_std
            result["upper"] = ensemble + ci_std_multiplier * pred_std

        return pd.DataFrame(result)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or MODELS_DIR / "ensemble_weights.pkl")
        joblib.dump({"weights": self.weights, "optimised": self.optimised_}, path)
        logger.info("Ensemble weights saved -> %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EnsembleForecaster":
        path = Path(path or MODELS_DIR / "ensemble_weights.pkl")
        data = joblib.load(path)
        obj  = cls(weights=data["weights"])
        obj.optimised_ = data["optimised"]
        logger.info("Ensemble weights loaded ← %s", path)
        return obj


# ── Stand-alone evaluation helper ────────────────────────────────────────────

def build_ensemble_forecast_df(
    dates:       pd.DatetimeIndex,
    preds_dict:  Dict[str, np.ndarray],
    actuals:     Optional[np.ndarray] = None,
    weights:     Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: build a final, dated forecast DataFrame.

    Args:
        dates      : business-day index for forecast horizon
        preds_dict : {model_name: array}
        actuals    : optional actual values (for evaluation)
        weights    : optional custom weights dict

    Returns:
        pd.DataFrame indexed by date
    """
    ens = EnsembleForecaster(weights=weights)
    df  = ens.predict(preds_dict)
    df.index = dates[: len(df)]
    if actuals is not None:
        df["actual"] = actuals[: len(df)]
    return df
