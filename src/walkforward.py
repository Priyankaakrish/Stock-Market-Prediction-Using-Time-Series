"""Walk-forward (rolling-origin) validation."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .config import PATHS
from .evaluate import directional_accuracy, mae, mape, rmse

log = logging.getLogger(__name__)


def make_folds(n_total: int, n_initial: int, step: int):
    """Expanding-window folds: train on [0, origin), test on the next block."""
    folds, origin = [], n_initial
    while origin < n_total:
        end = min(origin + step, n_total)
        if end - origin < max(step // 4, 5):
            break
        folds.append((slice(0, origin), slice(origin, end)))
        origin = end
    return folds


def _fresh(model):
    """Unfitted clone. Refitting in place would leak state between folds."""
    cls = type(model)
    for attr, kw in (("hp", None), ("params_", None), ("order", "order"),
                     ("window", "window"), ("refit_every", "refit_every")):
        if hasattr(model, attr):
            if kw is None:
                return cls(**getattr(model, attr))
            return cls(**{kw: getattr(model, attr)})
    return cls()


def walk_forward(features, models, n_initial, step=63, save=True):
    d = features.reset_index(drop=True)
    folds = make_folds(len(d), n_initial, step)
    log.info("Walk-forward: %d folds, initial=%d, step=%d", len(folds), n_initial, step)

    rows = []
    for i, (tr, te) in enumerate(folds):
        train, test = d.iloc[tr], d.iloc[te]
        cut = int(len(train) * 0.85)
        tr_fit, tr_val = train.iloc[:cut], train.iloc[cut:]
        for template in models:
            model = _fresh(template)
            t0 = time.time()
            try:
                model.fit(tr_fit, tr_val)
                if hasattr(model, "refit_on"):
                    model.refit_on(tr_fit, tr_val)
                pred = model.backtest(train, test)
            except Exception as err:
                log.warning("fold %d %s failed: %s", i, model.name, err)
                continue
            rows.append({
                "fold": i, "model": model.name,
                "test_start": test["Date"].iloc[0], "test_end": test["Date"].iloc[-1],
                "n_train": len(train), "n_test": len(test),
                "rmse": rmse(test["target"], pred),
                "mae": mae(test["target"], pred),
                "mape": mape(test["target"], pred),
                "directional_accuracy": directional_accuracy(
                    test["target"], pred, test["Close"]),
                "fit_seconds": time.time() - t0,
            })
        log.info("fold %d/%d done", i + 1, len(folds))

    df = pd.DataFrame(rows)
    if save and not df.empty:
        PATHS.ensure()
        df.to_csv(PATHS.reports / "walkforward_folds.csv", index=False)
    return df


def summarise(folds, baseline="naive"):
    """A model with lower mean RMSE but a 50% win rate is noisier, not better."""
    if folds.empty:
        return folds
    agg = (folds.groupby("model").agg(
        n_folds=("fold", "count"), rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"), rmse_worst=("rmse", "max"),
        mape_mean=("mape", "mean"),
        da_mean=("directional_accuracy", "mean"),
        da_std=("directional_accuracy", "std"),
    ).reset_index())

    if baseline in set(folds["model"]):
        base = folds[folds["model"] == baseline].set_index("fold")["rmse"]
        wins, skills = [], []
        for name in agg["model"]:
            r = folds[folds["model"] == name].set_index("fold")["rmse"]
            common = r.index.intersection(base.index)
            if name == baseline or len(common) == 0:
                wins.append(np.nan)
                skills.append(0.0 if name == baseline else np.nan)
                continue
            wins.append(float((r[common] < base[common]).mean() * 100.0))
            skills.append(float((1 - r[common] / base[common]).mean() * 100.0))
        agg["folds_beating_naive_pct"] = wins
        agg["mean_skill_vs_naive_pct"] = skills
    return agg.sort_values("rmse_mean").reset_index(drop=True)