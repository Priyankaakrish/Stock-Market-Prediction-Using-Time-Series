"""Stage 7: metrics and model comparison.

Beyond point-error metrics we compute two things that actually matter for a
trading-adjacent use case:

* **Directional accuracy** - did the model get the sign of tomorrow's move
  right? A model can have an excellent RMSE and 50% directional accuracy by
  simply echoing today's price.
* **Skill vs. the naive forecast** - the random-walk benchmark. On daily
  equity closes this is a genuinely hard baseline; any model that does not
  beat it has learned nothing useful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    model: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    r2: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask], mask


def rmse(y_true, y_pred) -> float:
    y_true, y_pred, _ = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred, _ = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred, _ = _clean(y_true, y_pred)
    nz = y_true != 0
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0)


def directional_accuracy(y_true, y_pred, last_close) -> float:
    """Fraction of days where sign(pred - today) == sign(actual - today)."""
    y_true, y_pred, mask = _clean(y_true, y_pred)
    last = np.asarray(last_close, dtype=float)[mask]
    true_dir = np.sign(y_true - last)
    pred_dir = np.sign(y_pred - last)
    valid = true_dir != 0
    if valid.sum() == 0:
        return float("nan")
    # The naive forecast predicts zero change every day, so it never commits to
    # a direction. Scoring that as 0% would be misleading; it is undefined.
    if np.all(pred_dir[valid] == 0):
        return float("nan")
    return float(np.mean(true_dir[valid] == pred_dir[valid]) * 100.0)


def r2(y_true, y_pred) -> float:
    y_true, y_pred, _ = _clean(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def evaluate(name: str, y_true, y_pred, last_close) -> Metrics:
    y_t, _y_p, _ = _clean(y_true, y_pred)
    return Metrics(
        model=name,
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
        directional_accuracy=directional_accuracy(y_true, y_pred, last_close),
        r2=r2(y_true, y_pred),
        n=len(y_t),
    )


def diebold_mariano(y_true, pred_a, pred_b, h: int = 1) -> tuple[float, float]:
    """Two-sided DM test on squared-error loss.

    Returns (statistic, p_value). Negative statistic => model A has lower loss.
    Uses a Newey-West variance with h-1 lags, plus the Harvey small-sample
    correction.
    """
    from scipy import stats

    y_true = np.asarray(y_true, float)
    d = (y_true - np.asarray(pred_a, float)) ** 2 - (y_true - np.asarray(pred_b, float)) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")

    d_bar = d.mean()
    gamma0 = np.sum((d - d_bar) ** 2) / n
    gamma = [
        np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n for k in range(1, h)
    ]
    var_d = (gamma0 + 2 * sum(gamma)) / n
    if var_d <= 0:
        return float("nan"), float("nan")

    dm = d_bar / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= correction
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def comparison_table(metrics: list[Metrics], baseline: str = "naive") -> pd.DataFrame:
    df = pd.DataFrame([m.as_dict() for m in metrics])
    if baseline in set(df["model"]):
        base_rmse = float(df.loc[df["model"] == baseline, "rmse"].iloc[0])
        # Positive skill => beats the random walk.
        df["skill_vs_naive_pct"] = (1 - df["rmse"] / base_rmse) * 100.0
    return df.sort_values("rmse").reset_index(drop=True)
