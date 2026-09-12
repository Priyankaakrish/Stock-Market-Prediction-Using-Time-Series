"""End-to-end training pipeline.

    ingest -> preprocess -> features -> split -> train -> evaluate
           -> MLflow tracking -> select best -> register -> export best_model.pkl

Run with ``python -m src.train``. Add ``--fast`` to skip Prophet and shrink the
LSTM (useful in CI).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import plots
from .config import DATA, MLFLOW, MODELS, PATHS
from .evaluate import Metrics, comparison_table, diebold_mariano, evaluate
from .features import build_features, feature_columns
from .ingest import ingest
from .models.baselines import DriftForecaster, MovingAverageForecaster, NaiveForecaster
from .preprocess import preprocess
from .split import chronological_split

log = logging.getLogger("train")

# Model key -> artefact stem, so saved files match the names in the data-flow
# diagram rather than the internal model keys.
ARTEFACT_NAMES = {"xgboost": "xgb"}


# --------------------------------------------------------------------------
def build_dataset(force_download: bool = False, reuse: bool = False):
    """Ingest -> clean -> features -> split.

    ``reuse=True`` loads the tables from disk instead of rebuilding them. The
    orchestrator in ``src/pipeline.py`` runs those stages itself and times them
    individually; without this it would hand control to a function that
    silently redid all four, doubling the work and reporting timings for
    stages that had already run.
    """
    if reuse and PATHS.features.exists():
        feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
        log.info("Reusing feature table (%d rows) from %s", len(feats), PATHS.features)
        return feats, chronological_split(feats, save=False)

    raw = ingest(force_download=force_download)
    cleaned = preprocess(raw)
    feats = build_features(cleaned)
    splits = chronological_split(feats)
    return feats, splits


def make_models(fast: bool = False, arima_order=None) -> list:
    from .models.arima_model import ARIMAForecaster
    from .models.lstm_model import LSTMForecaster
    from .models.xgb_model import XGBForecaster

    models = [
        NaiveForecaster(),
        MovingAverageForecaster(window=5),
        DriftForecaster(),
        ARIMAForecaster(order=arima_order or MODELS.arima_order),
        XGBForecaster(),
        LSTMForecaster(**({"epochs": 6, "patience": 3} if fast else {})),
    ]
    if not fast:
        try:
            from .models.prophet_model import ProphetForecaster
            models.insert(4, ProphetForecaster())
        except ImportError:
            log.warning("prophet not installed — skipping that model")
    return models


# --------------------------------------------------------------------------
def run(fast: bool = False, force_download: bool = False,
        reuse: bool = False) -> pd.DataFrame:
    import mlflow

    PATHS.ensure()
    t_start = time.time()

    feats, splits = build_dataset(force_download=force_download, reuse=reuse)
    print("\n" + splits.describe().to_string(index=False) + "\n")

    history = pd.concat([splits.train, splits.val], ignore_index=True)
    test = splits.test
    y_test = test["target"].to_numpy(dtype=float)
    last_close = test["Close"].to_numpy(dtype=float)

    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    if mlflow.get_experiment_by_name(MLFLOW.experiment) is None:
        mlflow.create_experiment(
            MLFLOW.experiment, artifact_location=MLFLOW.artifact_root)
    mlflow.set_experiment(MLFLOW.experiment)

    run_tag = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    all_metrics: list[Metrics] = []
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    run_ids: dict[str, str] = {}

    with mlflow.start_run(run_name=f"pipeline-{run_tag}"):
        mlflow.set_tags({
            "ticker": DATA.ticker,
            "horizon_days": DATA.horizon,
            "python": platform.python_version(),
            "pipeline": "nvda-timeseries",
        })
        mlflow.log_params({
            "n_rows": len(feats),
            "n_features": len(feature_columns(feats)),
            "train_frac": DATA.train_frac,
            "val_frac": DATA.val_frac,
            "train_end": str(splits.train["Date"].max().date()),
            "test_start": str(test["Date"].min().date()),
            "test_end": str(test["Date"].max().date()),
        })
        splits.describe().to_csv(PATHS.reports / "splits.csv", index=False)
        mlflow.log_artifact(str(PATHS.reports / "splits.csv"))
        mlflow.log_artifact(str(plots.plot_history_and_splits(splits)), "figures")

        arima_order = MODELS.arima_order
        if MODELS.arima_auto_order and not fast:
            from .models.arima_model import grid_search_order
            arima_order = grid_search_order(splits.train, splits.val)
            mlflow.log_param("arima_order_selected", str(arima_order))

        # Base rate of up-days: the number any directional accuracy must beat.
        up_rate = float((test["target"] > test["Close"]).mean() * 100.0)
        mlflow.log_metric("test_up_day_base_rate", up_rate)
        print(f"Test-period up-day base rate: {up_rate:.2f}%\n")

        for model in make_models(fast=fast, arima_order=arima_order):
            name = model.name
            log.info("=== %s ===", name)
            t0 = time.time()
            with mlflow.start_run(run_name=name, nested=True) as child:
                try:
                    model.fit(splits.train, splits.val)

                    # Validation score (model selection signal, not the headline).
                    val_pred = model.backtest(splits.train, splits.val)
                    val_m = evaluate(name, splits.val["target"], val_pred,
                                     splits.val["Close"])

                    # Tree model gets a final refit on train+val at tuned depth.
                    if hasattr(model, "refit_on"):
                        model.refit_on(splits.train, splits.val)

                    test_pred = model.backtest(history, test)
                    test_m = evaluate(name, y_test, test_pred, last_close)

                    fit_seconds = time.time() - t0
                    mlflow.log_params({**model.params(), "model_type": name})
                    mlflow.log_metrics({
                        "val_rmse": val_m.rmse, "val_mae": val_m.mae,
                        "val_mape": val_m.mape,
                        "val_directional_accuracy": val_m.directional_accuracy,
                        "test_rmse": test_m.rmse, "test_mae": test_m.mae,
                        "test_mape": test_m.mape,
                        "test_directional_accuracy": test_m.directional_accuracy,
                        "test_r2": test_m.r2,
                        "fit_seconds": fit_seconds,
                    })

                    all_metrics.append(test_m)
                    predictions[name] = test_pred
                    fitted[name] = model
                    run_ids[name] = child.info.run_id

                    # The data-flow diagram names these files explicitly
                    # (arima_model.pkl, prophet_model.pkl, xgb_model.pkl,
                    # lstm_model.pkl); keep the artefact names matching it.
                    artefact = PATHS.models / f"{ARTEFACT_NAMES.get(name, name)}_model.pkl"
                    model.save(artefact)
                    mlflow.log_artifact(str(artefact), "model")

                    if name == "xgboost":
                        imp = model.feature_importance(20)
                        imp.to_csv(PATHS.reports / "feature_importance.csv", index=False)
                        mlflow.log_artifact(str(PATHS.reports / "feature_importance.csv"))
                        mlflow.log_artifact(str(plots.plot_feature_importance(imp)), "figures")
                    if name == "lstm" and model.history_:
                        mlflow.log_artifact(
                            str(plots.plot_lstm_curve(model.history_)), "figures")

                    log.info("%-9s test RMSE=%.3f MAPE=%.2f%% DA=%.1f%% (%.1fs)",
                             name, test_m.rmse, test_m.mape,
                             test_m.directional_accuracy, fit_seconds)

                except Exception as err:
                    log.exception("Model %s failed: %s", name, err)
                    mlflow.set_tag("status", f"failed: {err}")

        # ---- comparison, significance, selection -------------------------
        table = comparison_table(all_metrics, baseline="naive")

        dm_rows = []
        if "naive" in predictions:
            for name, pred in predictions.items():
                if name == "naive":
                    continue
                stat, p = diebold_mariano(y_test, pred, predictions["naive"])
                dm_rows.append({"model": name, "dm_stat": stat, "p_value": p,
                                "beats_naive": bool(stat < 0 and p < 0.05)})
        dm = pd.DataFrame(dm_rows)
        if not dm.empty:
            table = table.merge(dm, on="model", how="left")

        table.to_csv(PATHS.metrics, index=False)
        mlflow.log_artifact(str(PATHS.metrics))

        pred_df = pd.DataFrame({"Date": test["Date"], "actual": y_test,
                                "last_close": last_close, **predictions})
        pred_df.to_csv(PATHS.predictions, index=False)
        mlflow.log_artifact(str(PATHS.predictions))

        mlflow.log_artifact(str(plots.plot_test_predictions(
            test["Date"], y_test, predictions)), "figures")
        mlflow.log_artifact(str(plots.plot_zoom(
            test["Date"], y_test, predictions)), "figures")
        mlflow.log_artifact(str(plots.plot_metric_comparison(table, up_rate)), "figures")

        # ---- pick the production model -----------------------------------
        # Business rule: minimise test RMSE, but require the model to be at
        # least as good as the random walk. If nothing beats the random walk we
        # say so loudly rather than shipping a model that adds no value.
        best_row = table.iloc[0]
        best_name = str(best_row["model"])
        naive_rmse = float(table.loc[table["model"] == "naive", "rmse"].iloc[0])

        best_da = table.sort_values("directional_accuracy", ascending=False).iloc[0]

        mlflow.log_metrics({
            "best_test_rmse": float(best_row["rmse"]),
            "naive_test_rmse": naive_rmse,
            "best_skill_vs_naive_pct": float(best_row.get("skill_vs_naive_pct", 0.0)),
        })
        mlflow.set_tags({
            "best_model": best_name,
            "best_directional_model": str(best_da["model"]),
            "beats_random_walk": str(best_name != "naive"),
        })

        if best_name in fitted:
            fitted[best_name].save(PATHS.best_model)
            meta = {
                "model": best_name,
                "trained_at": datetime.now(UTC).isoformat(),
                "horizon_days": DATA.horizon,
                "train_end": str(splits.train["Date"].max().date()),
                "test_start": str(test["Date"].min().date()),
                "test_end": str(test["Date"].max().date()),
                "metrics": {k: float(best_row[k]) for k in
                            ("rmse", "mae", "mape", "directional_accuracy", "r2")},
                "naive_rmse": naive_rmse,
                "mlflow_run_id": run_ids.get(best_name),
                "min_history_rows": getattr(fitted[best_name], "min_history", 1),
            }
            (PATHS.models / "best_model_meta.json").write_text(json.dumps(meta, indent=2))
            mlflow.log_artifact(str(PATHS.best_model), "best_model")
            mlflow.log_artifact(str(PATHS.models / "best_model_meta.json"), "best_model")

            # Register a real pyfunc model so the registry can stage/version it.
            try:
                from .mlflow_model import log_and_register

                example = pd.concat(
                    [history.tail(400), test.head(1)], ignore_index=True
                ).drop(columns=["target", "target_return"])
                log_and_register(str(PATHS.best_model), MLFLOW.registered_model,
                                 input_example=example.tail(2))
                log.info("Registered '%s' in the MLflow Model Registry",
                         MLFLOW.registered_model)
            except Exception as err:
                log.warning("Model registry unavailable (%s) — pickle still logged", err)

            resid_name = best_name if best_name != "naive" else (
                best_da["model"] if best_da["model"] != "naive" else best_name)
            mlflow.log_artifact(str(plots.plot_residual_diagnostics(
                y_test, predictions[resid_name], resid_name)), "figures")

    print("\n" + table.to_string(index=False))
    print(f"\nUp-day base rate in test window: {up_rate:.2f}% "
          f"(a permanently-bullish rule scores this)")
    print(f"Selected model: {best_name}")
    print(f"Best directional accuracy: {best_da['model']} "
          f"({best_da['directional_accuracy']:.2f}%)")
    print(f"Total pipeline time: {time.time() - t_start:.1f}s")
    return table


def run_walk_forward(step: int = 63, fast: bool = True) -> pd.DataFrame:
    """Rolling-origin validation across the test period, logged to MLflow.

    A single split gives one score per model; this gives a distribution, which
    is the only way to tell a real edge from a lucky window.
    """
    import mlflow
    from scipy import stats

    from .walkforward import summarise, walk_forward

    feats, splits = build_dataset()
    n_initial = len(splits.train) + len(splits.val)

    models = [m for m in make_models(fast=fast) if not (fast and m.name == "lstm")]
    folds = walk_forward(feats, models, n_initial=n_initial, step=step)
    summary = summarise(folds)

    # Per-fold win rate against the random walk. A mean RMSE can be dragged
    # around by one outlier fold; a win count cannot.
    base = folds[folds["model"] == "naive"].set_index("fold")["rmse"]
    rows = []
    for name in summary["model"]:
        if name == "naive":
            rows.append({"model": name, "wins": None, "binom_p": None})
            continue
        r = folds[folds["model"] == name].set_index("fold")["rmse"]
        common = r.index.intersection(base.index)
        wins = int((r[common] < base[common]).sum())
        rows.append({
            "model": name, "wins": wins,
            "binom_p": float(stats.binomtest(wins, len(common), 0.5).pvalue),
        })
    summary = summary.merge(pd.DataFrame(rows), on="model", how="left")
    summary.to_csv(PATHS.reports / "walkforward_summary.csv", index=False)

    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    mlflow.set_experiment(MLFLOW.experiment)
    with mlflow.start_run(run_name=f"walkforward-step{step}"):
        mlflow.log_params({"n_folds": int(folds["fold"].nunique()),
                           "step_days": step, "initial_train_rows": n_initial})
        for _, row in summary.iterrows():
            mlflow.log_metrics({
                f"wf_{row['model']}_rmse_mean": float(row["rmse_mean"]),
                f"wf_{row['model']}_rmse_std": float(row["rmse_std"]),
            })
            if pd.notna(row.get("binom_p")):
                mlflow.log_metric(f"wf_{row['model']}_binom_p", float(row["binom_p"]))
        mlflow.log_artifact(str(PATHS.reports / "walkforward_folds.csv"))
        mlflow.log_artifact(str(PATHS.reports / "walkforward_summary.csv"))
        mlflow.log_artifact(str(plots.plot_walkforward(folds, summary)), "figures")

    print("\n" + summary.round(3).to_string(index=False))
    print("\nA model that beats the random walk in roughly half the folds has no")
    print("edge; binom_p is the chance of that win count under a coin flip.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NVDA forecasting models")
    parser.add_argument("--fast", action="store_true",
                        help="skip Prophet, shrink LSTM (CI mode)")
    parser.add_argument("--download", action="store_true",
                        help="force a fresh yfinance download")
    parser.add_argument("--walk-forward", action="store_true",
                        help="rolling-origin validation instead of a single split")
    parser.add_argument("--step", type=int, default=63,
                        help="walk-forward block size in trading days (~63 = 1 quarter)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.walk_forward:
        run_walk_forward(step=args.step, fast=args.fast)
    else:
        run(fast=args.fast, force_download=args.download)


if __name__ == "__main__":
    main()
