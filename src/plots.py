"""Figures for the evaluation report. All saved under ``reports/figures``."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PATHS

NV_GREEN = "#76b900"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


def _save(fig, name: str) -> Path:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    path = PATHS.figures / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history_and_splits(splits) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    colors = {"train": "#3a7ca5", "val": "#f2a541", "test": NV_GREEN}
    for name, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        axes[0].plot(part["Date"], part["Close"], lw=0.9, color=colors[name],
                     label=f"{name} (n={len(part)})")
        axes[1].plot(part["Date"], part["log_ret"], lw=0.4, color=colors[name])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Close (log scale, USD)")
    axes[0].set_title("NVDA split-adjusted close — chronological splits")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set_ylabel("daily log return")
    axes[1].set_xlabel("Date")
    return _save(fig, "01_history_and_splits.png")


def plot_test_predictions(dates, actual, preds: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(dates, actual, color="black", lw=1.3, label="actual", zorder=5)
    for name, p in preds.items():
        axes[0].plot(dates, p, lw=0.9, alpha=0.8, label=name)
    axes[0].set_ylabel("Next-day close (USD)")
    axes[0].set_title("Test period — one-step-ahead forecasts")
    axes[0].legend(frameon=False, ncols=3, fontsize=8)

    for name, p in preds.items():
        axes[1].plot(dates, np.asarray(p, float) - np.asarray(actual, float),
                     lw=0.7, alpha=0.8, label=name)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("error (USD)")
    axes[1].set_xlabel("Date")
    return _save(fig, "02_test_predictions.png")


def plot_zoom(dates, actual, preds: dict[str, np.ndarray], last: int = 120) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates[-last:], actual[-last:], color="black", lw=1.6, marker="o",
            ms=2.5, label="actual")
    for name, p in preds.items():
        ax.plot(dates[-last:], np.asarray(p, float)[-last:], lw=1.1, alpha=0.85, label=name)
    ax.set_title(f"Last {last} test days (zoom)")
    ax.set_ylabel("Close (USD)")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    return _save(fig, "03_test_zoom.png")


def plot_metric_comparison(table: pd.DataFrame, up_day_base_rate: float | None = None) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    t = table.sort_values("rmse")

    # Log axis: Prophet's error is an order of magnitude larger than the rest,
    # which would flatten every other bar into invisibility on a linear scale.
    for ax, col, colour, label in (
        (axes[0], "rmse", NV_GREEN, "RMSE (USD, log scale)"),
        (axes[1], "mape", "#3a7ca5", "MAPE (%, log scale)"),
    ):
        ax.barh(t["model"], t[col], color=colour)
        ax.set_xscale("log")
        ax.set_title(label + " — lower is better")
        ax.invert_yaxis()
        for y, v in enumerate(t[col]):
            ax.text(v, y, f" {v:.2f}", va="center", fontsize=7.5)

    d = table.dropna(subset=["directional_accuracy"]).sort_values(
        "directional_accuracy", ascending=False)
    axes[2].barh(d["model"], d["directional_accuracy"], color="#f2a541")
    axes[2].axvline(50, color="red", ls="--", lw=1, label="coin flip (50%)")
    if up_day_base_rate is not None:
        axes[2].axvline(up_day_base_rate, color="black", ls=":", lw=1.4,
                        label=f"up-day base rate ({up_day_base_rate:.1f}%)")
    axes[2].set_title("Directional accuracy (%) — higher is better")
    axes[2].set_xlim(45, max(58, float(d["directional_accuracy"].max()) + 2))
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[2].invert_yaxis()
    return _save(fig, "04_model_comparison.png")


def plot_feature_importance(imp: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp["feature"], imp["importance"], color=NV_GREEN)
    ax.invert_yaxis()
    ax.set_title("XGBoost feature importance (gain-weighted)")
    return _save(fig, "05_feature_importance.png")


def plot_residual_diagnostics(actual, pred, name: str) -> Path:
    resid = np.asarray(pred, float) - np.asarray(actual, float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].hist(resid, bins=60, color=NV_GREEN, edgecolor="white")
    axes[0].set_title(f"{name}: residual distribution")
    axes[1].scatter(actual, resid, s=4, alpha=0.4, color="#3a7ca5")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("actual close")
    axes[1].set_title("residual vs level")
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid, lags=30, ax=axes[2], color="#3a7ca5")
    axes[2].set_title("residual ACF")
    return _save(fig, "06_residual_diagnostics.png")


def plot_lstm_curve(history: list[dict]) -> Path:
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h["epoch"], h["train_loss"], label="train", color="#3a7ca5")
    ax.plot(h["epoch"], h["val_loss"], label="validation", color=NV_GREEN)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (scaled returns)")
    ax.set_title("LSTM learning curve")
    ax.legend(frameon=False)
    return _save(fig, "07_lstm_learning_curve.png")
