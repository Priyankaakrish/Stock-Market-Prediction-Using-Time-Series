"""
pipelines/train_pipeline.py
─────────────────────────────
Orchestrates the full training pipeline:

  1.  Load & merge all NVDA data sources
  2.  Clean + feature-engineer (incl. scale-free features)
  3.  Temporal split (train / val / test)
  4.  Train ARIMA (log price), Prophet (log price), LSTM and XGBoost
      (next-day log return) — see src/forecasting.py for why
  5.  Evaluate every model one step ahead on validation and test sets,
      next to a naive "tomorrow = today" baseline
  6.  Optimise ensemble weights on the validation set
  7.  Log everything to MLflow
  8.  Refit ARIMA/Prophet on the full history and save all artefacts

Run:
    cd nvda_stock_prediction
    python pipelines/train_pipeline.py [--model all|arima|prophet|lstm|xgboost]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

# ── path fix: run from project root ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.data.loader import load_all_sources
from src.data.preprocessor import clean, temporal_split, DataScaler, prepare_data
from src.evaluation.metrics import regression_metrics, compare_models
from src.features.engineer import build_feature_set, get_model_feature_columns
from src.models.arima_model import ARIMAForecaster
from src.models.ensemble import EnsembleForecaster, build_ensemble_forecast_df
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import GBForecaster
from src.forecasting import (
    TARGET_RET, add_target, make_windows, lstm_logret, xgb_logret,
    returns_to_next_prices, save_feature_list,
)

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
import io as _io

# Force UTF-8 on stdout so Unicode chars don't crash on Windows cp1252 terminals
_stdout_handler = logging.StreamHandler(
    _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_stdout_handler.setFormatter(logging.Formatter(cfg.LOG_FORMAT))

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format=cfg.LOG_FORMAT,
    handlers=[
        _stdout_handler,
        logging.FileHandler(cfg.LOGS_DIR / "train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_pipeline")


# ── Step 1: Data ──────────────────────────────────────────────────────────────

def step_load_data() -> pd.DataFrame:
    logger.info("=== STEP 1: Load & Merge Data Sources ===")
    df_raw = load_all_sources()
    df_clean = clean(df_raw)
    df_feat  = build_feature_set(df_clean)
    logger.info("Feature matrix: %d rows × %d columns", *df_feat.shape)
    return df_feat


# ── Step 2: ARIMA ─────────────────────────────────────────────────────────────

def step_train_arima(df: pd.DataFrame, n_train: int) -> pd.Series:
    """Fit on train log-prices; one-step predictions for every later day."""
    logger.info("=== STEP 2: ARIMA (log price) ===")
    closes = df[cfg.TARGET_COL]
    arima = ARIMAForecaster(use_auto=True).fit_log(closes.iloc[:n_train])
    preds = arima.one_step_predictions(closes, start=n_train)
    mlflow.log_params({"arima_order": str(arima.order),
                       "arima_seasonal": str(arima.seasonal_order)})

    # Production model: same order, refitted on the most recent history
    final = ARIMAForecaster(order=arima.order,
                            seasonal_order=arima.seasonal_order, use_auto=False)
    final.fit_log(closes, reuse_order=True)
    final.save()
    return pd.Series(preds, index=df.index[n_train:])


# ── Step 3: Prophet ───────────────────────────────────────────────────────────

def step_train_prophet(df: pd.DataFrame, n_train: int, n_val: int) -> pd.Series:
    """
    Validation preds come from a model fitted on train; test preds from a model
    refitted on train+val (Prophet's trend needs the most recent data).
    """
    logger.info("=== STEP 3: Prophet (log price) ===")
    closes = df[cfg.TARGET_COL]
    out = []
    for fit_end, pred_end in [(n_train, n_train + n_val), (n_train + n_val, len(df))]:
        pf = ProphetForecaster().fit(df.iloc[:fit_end])
        window = closes.iloc[fit_end - 1: pred_end]       # include previous close
        out.append(pd.Series(pf.one_step_predictions(window), index=window.index[1:]))

    final = ProphetForecaster().fit(df)                  # production model
    final.save()
    return pd.concat(out)


# ── Step 4: LSTM ──────────────────────────────────────────────────────────────

def step_train_lstm(df, n_train, n_val, feat_cols) -> pd.Series:
    logger.info("=== STEP 4: Bidirectional LSTM (next-day return) ===")
    from src.models.lstm_model import BACKEND as LSTM_BACKEND
    if LSTM_BACKEND == "none":
        logger.warning("No deep-learning backend (install torch). LSTM skipped.")
        return pd.Series(dtype=float)

    scaler = DataScaler("standard")
    scaler.fit_transform(df.iloc[:n_train], feat_cols)
    scaler.save(cfg.MODELS_DIR / "feature_scaler.pkl")

    sc = scaler.transform(df, feat_cols)[feat_cols].values.astype("float32")
    sc = np.nan_to_num(np.clip(sc, -10, 10))
    target = df[TARGET_RET].values
    X, y, ends = make_windows(sc, target, cfg.LSTM_LOOKBACK)
    ok = ~np.isnan(y[:, 0])
    tr = ok & (ends < n_train - 1)          # target must stay inside train
    va = ok & (ends >= n_train) & (ends < n_train + n_val - 1)

    lf = LSTMForecaster(n_features=len(feat_cols))
    lf.build()
    history = lf.fit(X[tr], y[tr], X[va], y[va])
    mlflow.log_params({"lstm_lookback": cfg.LSTM_LOOKBACK,
                       "lstm_units": cfg.LSTM_UNITS,
                       "lstm_epochs_run": len(history["loss"])})
    lf.save()

    logret = lstm_logret(lf, df, feat_cols, scaler)
    return returns_to_next_prices(df, logret).iloc[n_train - 1:]


# ── Step 5: XGBoost ───────────────────────────────────────────────────────────

def step_train_xgboost(df, n_train, n_val, feat_cols) -> pd.Series:
    logger.info("=== STEP 5: XGBOOST (next-day return) ===")
    tr = df.iloc[:n_train - 1]                     # last train row's target is in val
    va = df.iloc[n_train: n_train + n_val - 1]
    gb = GBForecaster(model_type="xgboost", n_trials=cfg.XGB_N_TRIALS)
    gb.optimise(tr[feat_cols].values, tr[TARGET_RET].values, feature_names=feat_cols)
    gb.fit(tr[feat_cols].values, tr[TARGET_RET].values,
           va[feat_cols].values, va[TARGET_RET].values, feature_names=feat_cols)
    mlflow.log_params({"xgboost_best_params": str(gb.best_params_)})
    gb.save()
    logret = xgb_logret(gb, df, feat_cols)
    return returns_to_next_prices(df, logret).iloc[n_train - 1:]


# ── Step 6: Evaluation & ensemble ─────────────────────────────────────────────

def evaluate(preds: dict, dates: pd.DatetimeIndex, df: pd.DataFrame, split: str):
    """Metrics for every model on the given dates (one-step-ahead)."""
    actual = df[cfg.TARGET_COL].reindex(dates).values
    prev   = df[cfg.TARGET_COL].shift(1).reindex(dates).values
    table  = {}
    for name, s in preds.items():
        p = s.reindex(dates).values
        table[name] = regression_metrics(actual, p, label=f"{name} [{split}]",
                                         previous=prev)
        if name.startswith("naive"):      # flat forecast has no direction
            for k in ("Directional_Accuracy", "Sharpe_Ratio", "Max_Drawdown_pct"):
                table[name][k] = np.nan
        if split == "test":
            mlflow.log_metrics({f"{name}_{k}": v for k, v in table[name].items()
                                if np.isfinite(v)})
    return pd.DataFrame(table).T.round(4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(models_to_train: str = "all"):
    # Ensure tracking URI is always a valid file:// URI on Windows
    tracking_uri = cfg.MLFLOW_TRACKING_URI
    if not tracking_uri.startswith(("file://", "http://", "https://", "sqlite://",
                                     "postgresql://", "mysql://", "mssql://")):
        from urllib.request import pathname2url
        tracking_uri = "file:///" + pathname2url(str(tracking_uri)).lstrip("/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"full_pipeline_{models_to_train}") as run:
        mlflow.log_params({
            "ticker":         cfg.TICKER,
            "target_col":     cfg.TARGET_COL,
            "forecast_days":  cfg.FORECAST_DAYS,
            "train_ratio":    cfg.TRAIN_RATIO,
            "models":         models_to_train,
        })

        # ── Data & split ──────────────────────────────────────────────────────
        df = add_target(step_load_data())
        train_df, val_df, test_df = temporal_split(df)
        n_train, n_val = len(train_df), len(val_df)
        feat_cols = get_model_feature_columns(df)
        save_feature_list(feat_cols)
        logger.info("Model inputs: %d scale-free features", len(feat_cols))
        mlflow.log_params({
            "train_period": f"{train_df.index[0].date()}..{train_df.index[-1].date()}",
            "val_period":   f"{val_df.index[0].date()}..{val_df.index[-1].date()}",
            "test_period":  f"{test_df.index[0].date()}..{test_df.index[-1].date()}",
        })

        preds = {}
        run_it = lambda m: models_to_train in ("all", m)
        if run_it("arima"):
            preds["arima"] = step_train_arima(df, n_train)
        if run_it("prophet"):
            try:
                import prophet  # noqa: F401
                preds["prophet"] = step_train_prophet(df, n_train, n_val)
            except ImportError:
                logger.warning("Prophet not installed (pip install prophet) — skipping.")
        if run_it("lstm"):
            p = step_train_lstm(df, n_train, n_val, feat_cols)
            if len(p):
                preds["lstm"] = p
        if run_it("xgboost"):
            preds["xgboost"] = step_train_xgboost(df, n_train, n_val, feat_cols)

        # Naive baseline: tomorrow's close = today's close
        naive = df[cfg.TARGET_COL].shift(1)

        # ── Ensemble (weights fitted on validation) ───────────────────────────
        if len(preds) > 1:
            logger.info("=== STEP 6: Ensemble ===")
            ens = EnsembleForecaster()
            vd = val_df.index
            ens.fit_weights({k: v.reindex(vd).values for k, v in preds.items()},
                            val_df[cfg.TARGET_COL].values)
            ens.save()
            both = val_df.index.append(test_df.index)
            blended = ens.predict({k: v.reindex(both).values for k, v in preds.items()},
                                  compute_ci=False)["ensemble"].values
            preds["ensemble"] = pd.Series(blended, index=both)
            mlflow.log_params({"ensemble_weights": str(
                {k: round(v, 3) for k, v in ens.weights.items()})})

        preds["naive_last_close"] = naive

        # ── Evaluation ────────────────────────────────────────────────────────
        val_table  = evaluate(preds, val_df.index, df, "val")
        test_table = evaluate(preds, test_df.index, df, "test")
        logger.info("\nModel comparison — VALIDATION (%s .. %s):\n%s",
                    val_df.index[0].date(), val_df.index[-1].date(), val_table.to_string())
        logger.info("\nModel comparison — TEST (%s .. %s):\n%s",
                    test_df.index[0].date(), test_df.index[-1].date(), test_table.to_string())

        test_table.to_csv(cfg.DATA_PROC_DIR / "model_comparison.csv")
        val_table.to_csv(cfg.DATA_PROC_DIR / "model_comparison_val.csv")
        fc = pd.DataFrame({k: v.reindex(test_df.index) for k, v in preds.items()})
        fc.insert(0, "actual", test_df[cfg.TARGET_COL])
        fc.index.name = "Date"
        fc.to_csv(cfg.DATA_PROC_DIR / "test_forecast.csv")
        for f in ["model_comparison.csv", "model_comparison_val.csv", "test_forecast.csv"]:
            mlflow.log_artifact(str(cfg.DATA_PROC_DIR / f))

        logger.info("Training pipeline complete. Run ID: %s", run.info.run_id)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVDA Stock Prediction — Train Pipeline")
    parser.add_argument(
        "--model",
        choices=["all", "arima", "prophet", "lstm", "xgboost"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    args = parser.parse_args()
    sys.exit(main(args.model))
