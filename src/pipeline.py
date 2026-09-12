"""Offline pipeline orchestrator.

Every stage in the architecture's offline band existed as a module, but nothing
ran them in the order the arrows describe. ``src/train.py`` goes straight from
ingestion to pandas feature engineering and never touches S3 or Spark, so the
data-lake and EMR boxes were reachable only by hand.

This is the connective tissue. Three modes:

``local``   ingest -> pandas features -> train. The default, and what runs in
            CI: no AWS, no JVM, ~3 minutes.

``s3``      ingest -> land raw in S3 -> pandas features -> publish processed
            partitions -> train. The lake is populated; compute stays local.
            This is the shape for a single ticker, where Spark's overhead
            exceeds the work.

``spark``   ingest -> land raw -> **EMR/Spark features** -> publish -> train.
            The full diagram path. Worth the JVM only when the feature table
            outgrows one machine — a universe of tickers, or intraday bars.

The modes share one contract: whatever produces the feature table, training
consumes ``data/processed/nvda_features.csv``. That is what makes the Spark
path substitutable rather than a parallel universe, and why
``--validate-against`` exists to prove the two agree before one replaces
the other.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA, PATHS

log = logging.getLogger(__name__)


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float
    detail: str = ""
    outputs: list[str] = field(default_factory=list)


class PipelineError(RuntimeError):
    """A stage failed and the run cannot meaningfully continue."""


def _timed(name: str, fn, *args, **kwargs) -> StageResult:
    t0 = time.time()
    try:
        detail, outputs = fn(*args, **kwargs)
        r = StageResult(name, True, time.time() - t0, detail, outputs)
        log.info("%-22s ok    %5.1fs  %s", name, r.seconds, detail)
        return r
    except Exception as err:
        r = StageResult(name, False, time.time() - t0, f"{type(err).__name__}: {err}")
        log.error("%-22s FAIL  %5.1fs  %s", name, r.seconds, r.detail)
        raise PipelineError(f"{name}: {err}") from err


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def _stage_ingest(force_download: bool):
    from .ingest import ingest

    df = ingest(force_download=force_download)
    return (f"{len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}",
            [str(PATHS.raw)])


def _stage_land_raw(bucket: str | None, fmt: str):
    from .storage import upload_raw

    uri = upload_raw(bucket=bucket, fmt=fmt)
    return f"landed as {fmt}", [uri]


def _stage_preprocess():
    """Its own stage, as the data-flow diagram draws it.

    Cleaning and feature engineering fail differently and are worth timing
    separately: this one is where duplicate dates, non-positive prices and
    zero-volume halts get resolved, and where the decision not to reindex onto
    a calendar grid lives.
    """
    from .preprocess import preprocess

    cleaned = preprocess()
    weekday_share = (cleaned["Date"].dt.dayofweek < 5).mean() * 100
    return (f"{len(cleaned)} rows, {weekday_share:.1f}% weekdays "
            "(no calendar reindex)", [str(PATHS.cleaned)])


def _stage_features_pandas():
    """Time-series feature engineering.

    Reports the split between model-facing and reference columns, because the
    gap is the point: raw price levels stay in the table for plotting and
    reconstruction but never reach a model.
    """
    import pandas as pd

    from .features import build_features, feature_columns

    cleaned = pd.read_csv(PATHS.cleaned, parse_dates=["Date"])
    feats = build_features(cleaned)
    model_cols = feature_columns(feats)
    groups = {
        "MA ratios": sum(c.startswith("close_over_MA") or "_over_MA" in c
                         for c in model_cols),
        "oscillators": sum(c.startswith(("RSI", "MACD", "BB_")) for c in model_cols),
        "lags": sum("lag_" in c for c in model_cols),
        "volatility": sum("volatil" in c or "ATR" in c for c in model_cols),
    }
    breakdown = " ".join(f"{k}={v}" for k, v in groups.items())
    return (f"{len(feats)} rows, {len(model_cols)}/{feats.shape[1]} columns "
            f"model-facing; {breakdown}",
            [str(PATHS.features)])


def _stage_features_spark(spark_output: str, validate: bool):
    """Run the EMR job, then hand its output to the same downstream contract.

    ``--validate-against`` is not optional here: swapping the feature
    implementation under a trained model without proving parity is how the two
    paths silently diverge.
    """
    script = Path(__file__).parent / "spark_jobs" / "prepare_data_spark.py"
    cmd = [sys.executable, str(script),
           "--input", str(PATHS.raw), "--output", spark_output,
           "--partition-by", "year-month"]
    if validate:
        if not PATHS.features.exists():
            raise PipelineError(
                "Parity check needs the pandas feature table. Run the local "
                "pipeline once first, or pass --no-validate to skip.")
        cmd += ["--validate-against", str(PATHS.features)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise PipelineError(f"Spark job failed:\n{tail}")

    passed = "Parity check passed" in proc.stdout
    detail = "parity verified" if passed else "written (parity not checked)"
    return detail, [spark_output]


def _stage_publish_processed(bucket: str | None):
    from .storage import upload_processed

    uris = upload_processed(bucket=bucket)
    return f"{len(uris)} year/month partitions", uris[:3] + (["..."] if len(uris) > 3 else [])


def _stage_split():
    """Chronological train/validation/test split.

    Its own stage because it is where leakage enters, and because the numbers
    it reports are the ones that explain every result downstream: training
    tops out near $6 while the test window reaches $207, which is why the
    tree and network models learn returns rather than price levels.

    The no-leakage guarantee is asserted here rather than assumed —
    train.max < val.min < test.min, checked on every run.
    """
    import pandas as pd

    from .split import chronological_split

    feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
    splits = chronological_split(feats, save=True)

    # The property the whole evaluation rests on.
    assert splits.train["Date"].max() < splits.val["Date"].min()
    assert splits.val["Date"].max() < splits.test["Date"].min()

    tr, va, te = splits.train, splits.val, splits.test
    ratio = te["Close"].max() / tr["Close"].max()
    detail = (f"{len(tr)}/{len(va)}/{len(te)} rows, "
              f"test {te['Date'].min().date()}..{te['Date'].max().date()}, "
              f"close ${tr['Close'].max():.2f} -> ${te['Close'].max():.2f} "
              f"({ratio:.0f}x), no leakage")
    return detail, [str(PATHS.train), str(PATHS.val), str(PATHS.test)]


def _stage_walk_forward(step: int, fast: bool):
    """Rolling-origin validation.

    The diagram puts walk-forward inside the Train/Validation/Test box, not
    beside it: a chronological split alone gives one score per model and cannot
    tell a real edge from a lucky window. Running it as part of the pipeline is
    what makes the single-split table interpretable.
    """
    from scipy import stats

    from .walkforward import summarise, walk_forward

    feats, splits = _feature_table()
    n_initial = len(splits.train) + len(splits.val)
    models = _wf_models(fast)

    folds = walk_forward(feats, models, n_initial=n_initial, step=step)
    summary = summarise(folds)

    base = folds[folds["model"] == "naive"].set_index("fold")["rmse"]
    verdicts = []
    for name in summary["model"]:
        if name == "naive":
            continue
        r = folds[folds["model"] == name].set_index("fold")["rmse"]
        common = r.index.intersection(base.index)
        wins = int((r[common] < base[common]).sum())
        p = float(stats.binomtest(wins, len(common), 0.5).pvalue)
        verdicts.append(f"{name} {wins}/{len(common)} p={p:.3f}")

    summary.to_csv(PATHS.reports / "walkforward_summary.csv", index=False)
    return (f"{folds['fold'].nunique()} folds; " + ", ".join(verdicts),
            [str(PATHS.reports / "walkforward_folds.csv"),
             str(PATHS.reports / "walkforward_summary.csv")])


def _feature_table():
    import pandas as pd

    from .split import chronological_split

    feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
    return feats, chronological_split(feats, save=False)


def _wf_models(fast: bool):
    """Models cheap enough to refit 16 times. Prophet needs ~16 Stan fits and
    the LSTM 16 training runs, so both are opt-in."""
    from .models.arima_model import ARIMAForecaster
    from .models.baselines import DriftForecaster, MovingAverageForecaster, NaiveForecaster
    from .models.xgb_model import XGBForecaster

    models = [NaiveForecaster(), MovingAverageForecaster(5), DriftForecaster(),
              XGBForecaster()]
    if not fast:
        models.insert(3, ARIMAForecaster())
    return models


def _stage_train(fast: bool, reuse: bool = True):
    """Model training + baseline evaluation.

    The diagram draws these as two boxes and they are two ideas, but one
    function: every model is scored by the same harness on the same split, so
    separating them would mean running the harness twice. What is reported here
    is the comparison, not the winner — a champion without its baseline is the
    number this whole project exists to distrust.
    """
    from .train import run

    table = run(fast=fast, reuse=reuse)
    best = table.iloc[0]
    naive = table.loc[table["model"] == "naive", "rmse"].iloc[0]

    ts_models = [m for m in ("arima", "prophet", "xgboost", "lstm")
                 if m in set(table["model"])]
    baselines = [m for m in ("naive", "ma5", "drift") if m in set(table["model"])]
    beat = int((table["rmse"] < naive).sum())

    return (f"time-series: {'/'.join(ts_models)}; baselines: {'/'.join(baselines)}; "
            f"best={best['model']} {best['rmse']:.4f} vs naive {naive:.4f}; "
            f"{beat} of {len(table) - 1} beat naive",
            [str(PATHS.best_model), str(PATHS.metrics)])


def _stage_verify_serving():
    """Resolve the Production alias and make one real prediction.

    Promotion moves a pointer; it does not prove the thing pointed at can
    serve. This project shipped a broken serving path for its entire life
    because nothing ever resolved ``models:/nvda-forecaster@Production`` —
    MLflow logged the calendar features as int32 and schema enforcement
    rejected every int64 input, while the API's fallback to the local pickle
    kept the failure invisible.

    One inference through the registry is the smallest check that would have
    caught it, so it runs on every promotion.
    """
    import mlflow
    import pandas as pd

    from .config import MLFLOW

    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    uri = f"models:/{MLFLOW.registered_model}@Production"
    model = mlflow.pyfunc.load_model(uri)

    feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
    window = feats.tail(400).drop(columns=["target", "target_return"])
    out = model.predict(window)

    predicted = float(out["predicted_close"].iloc[0])
    last = float(out["last_close"].iloc[0])
    move = (predicted / last - 1) * 100

    # A one-day forecast beyond this is not a model, it is a fault.
    if abs(move) > 25:
        raise PipelineError(
            f"Production model predicts a {move:.1f}% daily move — refusing to "
            "call this a working deployment.")

    # The registry answering is necessary but not sufficient: the API has its
    # own code path for pyfunc models, and that path shipped broken. Exercise
    # it here so "verified" means the service can serve, not just the registry.
    import os

    from fastapi.testclient import TestClient

    prev = os.environ.get("MODEL_URI")
    os.environ["MODEL_URI"] = uri
    try:
        import importlib

        import api.main as api_main
        importlib.reload(api_main)
        with TestClient(api_main.app) as client:
            resp = client.post("/forecast", json={})
        if resp.status_code != 200:
            raise PipelineError(
                f"API cannot serve from {uri}: HTTP {resp.status_code} "
                f"{resp.text[:200]}")
        api_price = resp.json()["predicted_close"]
    finally:
        if prev is None:
            os.environ.pop("MODEL_URI", None)
        else:
            os.environ["MODEL_URI"] = prev

    # Not compared for equality: the registry call above forecasts from the
    # feature table, whose last row is the last day that *has a label*, while
    # the API pads its cached bars so the true final bar survives. They
    # legitimately forecast different days. What matters is that both produce a
    # plausible number through the same model.
    api_move = abs(api_price / resp.json()["last_close"] - 1) * 100
    if api_move > 25:
        raise PipelineError(
            f"API predicts a {api_move:.1f}% daily move from {uri}")

    return (f"{uri} -> {predicted:.4f} from {last:.4f} ({move:+.3f}%); "
            f"API serves {api_price:.4f} ({api_move:+.3f}%)", [uri])


def _stage_promote():
    from .promote import promote

    result = promote()
    return ("approved" if result.approved else "REFUSED by the gate",
            [f"models:/nvda-forecaster@Production -> v{result.version}"])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline(
    mode: str = "local",
    bucket: str | None = None,
    fast: bool = False,
    force_download: bool = False,
    raw_fmt: str = "csv",
    spark_output: str = "data/processed/spark",
    validate_spark: bool = True,
    do_promote: bool = False,
    walk_forward_step: int = 0,
) -> list[StageResult]:
    """Run the offline band end to end. Returns one result per stage."""
    if mode not in ("local", "s3", "spark"):
        raise ValueError("mode must be local, s3 or spark")

    log.info("=" * 64)
    log.info("Offline pipeline: mode=%s ticker=%s", mode, DATA.ticker)
    log.info("=" * 64)

    results = [_timed("ingest", _stage_ingest, force_download)]

    if mode in ("s3", "spark"):
        results.append(_timed("land raw -> S3", _stage_land_raw, bucket, raw_fmt))

    if mode == "spark":
        results.append(_timed("features (EMR/Spark)", _stage_features_spark,
                              spark_output, validate_spark))
    else:
        results.append(_timed("preprocess", _stage_preprocess))
        results.append(_timed("feature engineering", _stage_features_pandas))

    if mode in ("s3", "spark"):
        results.append(_timed("publish processed", _stage_publish_processed, bucket))

    results.append(_timed("train/val/test split", _stage_split))

    results.append(_timed("train + evaluate", _stage_train, fast))

    if walk_forward_step:
        results.append(_timed("walk-forward validation", _stage_walk_forward,
                              walk_forward_step, fast))

    if do_promote:
        results.append(_timed("approval gate", _stage_promote))
        results.append(_timed("verify serving path", _stage_verify_serving))

    total = sum(r.seconds for r in results)
    log.info("=" * 64)
    log.info("%d stages, %.1fs total", len(results), total)
    for r in results:
        for o in r.outputs:
            log.info("    %-22s %s", r.name, o)
    log.info("=" * 64)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the offline data & ML pipeline end to end")
    ap.add_argument("--mode", default="local", choices=["local", "s3", "spark"],
                    help="local: no AWS. s3: populate the lake. spark: full EMR path.")
    ap.add_argument("--bucket", default=None, help="overrides S3_BUCKET")
    ap.add_argument("--fast", action="store_true", help="skip Prophet, shrink LSTM")
    ap.add_argument("--download", action="store_true", help="force a fresh yfinance pull")
    ap.add_argument("--raw-format", default="csv", choices=["csv", "parquet"])
    ap.add_argument("--spark-output", default="data/processed/spark")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the Spark/pandas parity check (not recommended)")
    ap.add_argument("--walk-forward", type=int, nargs="?", const=63, default=0,
                    metavar="STEP",
                    help="run rolling-origin validation with STEP-day folds "
                         "(default 63, ~one trading quarter)")
    ap.add_argument("--promote", action="store_true",
                    help="run the approval gate and set the Production alias")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        run_pipeline(mode=args.mode, bucket=args.bucket, fast=args.fast,
                     force_download=args.download, raw_fmt=args.raw_format,
                     spark_output=args.spark_output,
                     validate_spark=not args.no_validate, do_promote=args.promote,
                     walk_forward_step=args.walk_forward)
    except PipelineError as err:
        log.error("Pipeline aborted. %s", err)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
