"""
src/evaluation/metrics.py
──────────────────────────
Comprehensive regression and finance-specific evaluation metrics.

Metrics:
  Standard  : MAE, RMSE, MAPE, R², Explained Variance
  Finance   : Directional Accuracy, Sharpe ratio of strategy returns,
              Maximum Drawdown, Hit Rate, Information Coefficient (IC)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Core regression metrics ───────────────────────────────────────────────────

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — bounded between 0 and 200 %."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def explained_variance(actual: np.ndarray, predicted: np.ndarray) -> float:
    residuals = actual - predicted
    if actual.var() == 0:
        return np.nan
    return float(1 - residuals.var() / actual.var())


# ── Finance-specific metrics ──────────────────────────────────────────────────

def directional_accuracy(actual: np.ndarray, predicted: np.ndarray,
                         previous: Optional[np.ndarray] = None) -> float:
    """
    Percentage of time-steps where the predicted direction
    matches the actual direction (up/down).

    If `previous` (the last known close before each prediction) is given, the
    direction is measured against it — the correct test for one-step-ahead
    forecasts. Otherwise consecutive predictions are compared.
    """
    if previous is not None:
        previous      = np.asarray(previous, dtype=float)[: len(actual)]
        actual_dir    = np.sign(actual - previous)
        predicted_dir = np.sign(predicted - previous)
        return float((actual_dir == predicted_dir).mean() * 100)
    actual_dir    = np.sign(np.diff(actual))
    predicted_dir = np.sign(np.diff(predicted))
    correct       = (actual_dir == predicted_dir).sum()
    total         = len(actual_dir)
    return float(correct / total * 100) if total > 0 else np.nan


def strategy_returns(
    actual:     np.ndarray,
    predicted:  np.ndarray,
    transaction_cost: float = 0.001,   # 10 bps one-way
    previous: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Simple long/flat strategy returns based on prediction direction.
    Goes long if predicted return > 0, otherwise stays flat.
    """
    if previous is not None:
        previous   = np.asarray(previous, dtype=float)[: len(actual)]
        actual_ret = actual / previous - 1
        pred_dir   = (predicted > previous).astype(float)
    else:
        actual_ret = np.diff(actual) / actual[:-1]
        pred_dir   = (np.diff(predicted) > 0).astype(float)
    trades     = np.abs(np.diff(pred_dir, prepend=0.0))
    strat_ret  = pred_dir * actual_ret - transaction_cost * trades
    return pd.Series(strat_ret, name="strategy_returns")


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate assumed 0)."""
    if returns.std() == 0:
        return np.nan
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of a cumulative return series."""
    cum    = (1 + returns).cumprod()
    peak   = cum.expanding().max()
    dd     = (cum - peak) / peak
    return float(dd.min() * 100)   # in %


def information_coefficient(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Rank correlation between actual and predicted returns.
    IC > 0.05 is generally considered good in quantitative finance.
    """
    from scipy.stats import spearmanr
    if len(actual) < 3:
        return np.nan
    ic, _ = spearmanr(actual, predicted)
    return float(ic)


# ── Master evaluation function ────────────────────────────────────────────────

def regression_metrics(
    actual:    np.ndarray,
    predicted: np.ndarray,
    label:     str = "model",
    include_finance: bool = True,
    previous:  Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute and log all metrics.

    Returns a flat dict of metric_name -> value.
    """
    actual    = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()
    n         = min(len(actual), len(predicted))
    actual    = actual[:n]
    predicted = predicted[:n]
    if previous is not None:
        previous = np.asarray(previous, dtype=float).flatten()[:n]

    metrics = {
        "MAE":               mae(actual, predicted),
        "RMSE":              rmse(actual, predicted),
        "MAPE":              mape(actual, predicted),
        "SMAPE":             smape(actual, predicted),
        "R2":                r_squared(actual, predicted),
        "Explained_Variance":explained_variance(actual, predicted),
    }

    if include_finance and n > 1:
        metrics["Directional_Accuracy"] = directional_accuracy(actual, predicted, previous)
        try:
            strat = strategy_returns(actual, predicted, previous=previous)
            metrics["Sharpe_Ratio"]   = sharpe_ratio(strat)
            metrics["Max_Drawdown_pct"] = max_drawdown(strat)
        except Exception:
            pass
        metrics["IC"] = information_coefficient(actual, predicted)

    # Log
    logger.info("=== %s Metrics ===", label)
    for k, v in metrics.items():
        logger.info("  %-25s %.4f", k, v)

    return metrics


def compare_models(
    actuals:     np.ndarray,
    preds_dict:  Dict[str, np.ndarray],
    previous:    Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build a comparison table of all models' metrics.

    Returns a DataFrame with metrics as rows and models as columns.
    """
    rows = {}
    for name, preds in preds_dict.items():
        rows[name] = regression_metrics(actuals, preds, label=name, previous=previous)
    df = pd.DataFrame(rows).T
    return df.round(4)
