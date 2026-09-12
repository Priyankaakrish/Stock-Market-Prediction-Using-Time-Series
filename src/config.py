"""Central configuration for the NVDA forecasting pipeline.

Every path and hyper-parameter lives here so that the training pipeline, the
API and the tests all agree on the same contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = _p("data", "raw", "nvda_raw.csv")
    cleaned: Path = _p("data", "processed", "nvda_cleaned.csv")
    features: Path = _p("data", "processed", "nvda_features.csv")
    train: Path = _p("data", "processed", "nvda_train.csv")
    val: Path = _p("data", "processed", "nvda_val.csv")
    test: Path = _p("data", "processed", "nvda_test.csv")
    models: Path = _p("models")
    best_model: Path = _p("models", "best_model.pkl")
    metrics: Path = _p("reports", "metrics.csv")
    predictions: Path = _p("reports", "test_predictions.csv")
    figures: Path = _p("reports", "figures")
    reports: Path = _p("reports")

    def ensure(self) -> None:
        for d in (
            self.raw.parent,
            self.cleaned.parent,
            self.models,
            self.figures,
            self.reports,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    ticker: str = "NVDA"
    start: str = "1999-01-01"
    end: str | None = None                  # None -> today
    target_col: str = "Close"
    horizon: int = 1                        # forecast t+1
    train_frac: float = 0.70
    val_frac: float = 0.15                  # test gets the remainder
    ma_windows: tuple[int, ...] = (7, 20, 50)
    lags: tuple[int, ...] = (1, 2, 3, 5, 7, 14, 30)
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd: tuple[int, int, int] = (12, 26, 9)
    vol_windows: tuple[int, ...] = (7, 21)


@dataclass(frozen=True)
class ModelConfig:
    # ARIMA is fitted on log(price); d=1 makes it a model of log-returns.
    arima_order: tuple[int, int, int] = (2, 1, 2)
    arima_auto_order: bool = True          # pick (p,d,q) by AIC on the train fold

    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 600,
            "learning_rate": 0.02,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 20,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 4,
        }
    )

    lstm_params: dict = field(
        default_factory=lambda: {
            "lookback": 30,
            "hidden_size": 48,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "epochs": 40,
            "batch_size": 64,
            "patience": 6,
            "seed": 42,
        }
    )

    # Prophet is a trend/seasonality model: refitting it every single day over a
    # 1000-day test window is not realistic in production, so we retrain on a
    # fixed cadence and forecast forward until the next retrain.
    prophet_refit_every: int = 63          # ~one trading quarter
    prophet_params: dict = field(
        default_factory=lambda: {
            "daily_seasonality": False,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.10,
            "seasonality_mode": "additive",
            "interval_width": 0.80,
        }
    )


@dataclass(frozen=True)
class MLflowConfig:
    # MLflow 3.x put the plain-file store into maintenance mode, so the local
    # default is SQLite — which is also what the EBS-backed MLflow server in
    # the deployment diagram uses. Point MLFLOW_TRACKING_URI at the remote
    # server (http://<host>:5002) in production.
    tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"
    )
    artifact_root: str = os.getenv("MLFLOW_ARTIFACT_ROOT", str(ROOT / "mlartifacts"))
    experiment: str = os.getenv("MLFLOW_EXPERIMENT", "nvda-price-forecasting")
    registered_model: str = "nvda-forecaster"
    # Metric used to pick the production model. Lower is better.
    selection_metric: str = "test_rmse"


PATHS = Paths()
DATA = DataConfig()
MODELS = ModelConfig()
MLFLOW = MLflowConfig()

# Columns that must never be used as predictors: they contain information from
# time t+1 or are the target itself in disguise.
LEAKY_COLUMNS = {"target", "target_return", "Date", "Adj Close"}


def load_secret(name: str, key: str | None = None, region: str | None = None):
    """Fetch a secret from AWS Secrets Manager.

    The diagram lists Secrets Manager under Security & Access, and
    ``deploy/aws/iam_policy.json`` already grants ``secretsmanager:GetSecretValue``
    scoped to ``nvda/*``. This is the code behind that grant.

    Nothing in the current pipeline needs a secret — the data source is public
    and AWS access comes from the instance profile. It exists for the moment one
    does (a paid market-data key, a database password), so that the answer is
    not "put it in an environment variable on the box".

    Returns None rather than raising when unavailable: a missing optional
    secret should not take the API down at startup.
    """
    import json as _json

    try:
        import boto3
    except ImportError:
        log_msg = "boto3 not installed; cannot read secret %s"
        import logging as _logging
        _logging.getLogger(__name__).warning(log_msg, name)
        return None

    try:
        client = boto3.client(
            "secretsmanager",
            region_name=region or os.getenv("AWS_REGION", "ap-south-1"))
        payload = client.get_secret_value(SecretId=name)["SecretString"]
    except Exception as err:
        import logging as _logging
        _logging.getLogger(__name__).warning("Secret %s unavailable: %s", name, err)
        return None

    try:
        parsed = _json.loads(payload)
    except _json.JSONDecodeError:
        return payload                     # plain string secret
    return parsed.get(key) if key else parsed
