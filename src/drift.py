"""Data drift and model drift detection.

A deployed forecaster fails silently: the API keeps returning 200s and
plausible prices long after the world moved away from its training data.

Two distinct questions, often conflated:

Data drift  - have the *inputs* changed? Compared against the training
distribution with Population Stability Index and a KS test. Needs no labels,
so it is available the moment new bars arrive.

Model drift - has *accuracy* degraded? Needs realised outcomes, so it lags by
the forecast horizon. Measured against the baseline AND against naive on the
same window. A model whose RMSE rose because the stock got more volatile has
not degraded; one that lost ground relative to the random walk has.

PSI thresholds follow the credit-risk convention: <0.10 stable, 0.10-0.25
moderate, >0.25 significant. They are heuristics, which is why the KS p-value
is reported beside them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .config import PATHS
from .evaluate import rmse

log = logging.getLogger(__name__)

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    mean_shift_sigma: float
    severity: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    checked_at: str
    n_reference: int
    n_current: int
    current_start: str
    current_end: str
    n_features: int
    n_moderate: int
    n_significant: int
    max_psi: float
    worst_feature: str
    data_drift_detected: bool
    features: list = field(default_factory=list)
    model_drift: dict | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["features"] = [f.as_dict() if isinstance(f, FeatureDrift) else f
                         for f in self.features]
        return d


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI between two samples, using quantile bins from the reference.

    Quantile rather than equal-width bins: vol_over_MA20 is heavily
    right-skewed, and equal-width bins would put 95% of reference mass in one
    bucket and report PSI ~0 no matter what happened.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins * 2 or len(current) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)

    # Floor: an empty bucket would otherwise send PSI to infinity.
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _severity(psi: float) -> str:
    if not np.isfinite(psi):
        return "unknown"
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def detect_feature_drift(reference, current, features=None):
    from scipy import stats

    if features is None:
        from .features import feature_columns
        features = feature_columns(reference)

    out = []
    for col in features:
        if col not in current.columns:
            continue
        ref = reference[col].to_numpy(dtype=float)
        cur = current[col].to_numpy(dtype=float)
        ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        if len(ref) < 30 or len(cur) < 10:
            continue

        psi = population_stability_index(ref, cur)
        ks_stat, ks_p = stats.ks_2samp(ref, cur)
        sd = ref.std() or 1.0
        out.append(FeatureDrift(
            feature=col,
            psi=round(psi, 5),
            ks_statistic=round(float(ks_stat), 5),
            ks_pvalue=round(float(ks_p), 6),
            reference_mean=round(float(ref.mean()), 6),
            current_mean=round(float(cur.mean()), 6),
            mean_shift_sigma=round(float((cur.mean() - ref.mean()) / sd), 4),
            severity=_severity(psi),
        ))
    return sorted(out, key=lambda f: (-f.psi if np.isfinite(f.psi) else 0))


def detect_model_drift(recent, predictions, baseline_rmse, tolerance=1.25) -> dict:
    """Compare recent accuracy to the baseline AND to naive on the same window.

    The naive comparison is the important one: dollar RMSE rises here simply
    because the price level rises. Only the ratio to the random walk shows
    whether the model itself decayed.
    """
    actual = recent["target"].to_numpy(dtype=float)
    last_close = recent["Close"].to_numpy(dtype=float)
    current = rmse(actual, predictions)
    naive = rmse(actual, last_close)

    ratio_baseline = current / baseline_rmse if baseline_rmse else float("nan")
    ratio_naive = current / naive if naive else float("nan")
    return {
        "n_observations": len(recent),
        "window_start": str(pd.Timestamp(recent["Date"].iloc[0]).date()),
        "window_end": str(pd.Timestamp(recent["Date"].iloc[-1]).date()),
        "current_rmse": round(current, 5),
        "naive_rmse_same_window": round(naive, 5),
        "baseline_rmse": round(float(baseline_rmse), 5),
        "rmse_vs_baseline": round(ratio_baseline, 4),
        "rmse_vs_naive": round(ratio_naive, 4),
        "degraded_vs_baseline": bool(ratio_baseline > tolerance),
        "worse_than_naive": bool(ratio_naive > 1.0),
    }


def build_report(reference, current, model=None, baseline_rmse=None, save=True):
    feats = detect_feature_drift(reference, current)
    n_sig = sum(f.severity == "significant" for f in feats)
    n_mod = sum(f.severity == "moderate" for f in feats)
    worst = feats[0] if feats else None

    model_drift = None
    if model is not None and baseline_rmse is not None and "target" in current:
        try:
            preds = model.backtest(reference, current)
            model_drift = detect_model_drift(current, preds, baseline_rmse)
        except Exception as err:
            log.warning("Model drift check failed: %s", err)

    report = DriftReport(
        checked_at=datetime.now(UTC).isoformat(),
        n_reference=len(reference),
        n_current=len(current),
        current_start=str(pd.Timestamp(current["Date"].iloc[0]).date()),
        current_end=str(pd.Timestamp(current["Date"].iloc[-1]).date()),
        n_features=len(feats),
        n_moderate=n_mod,
        n_significant=n_sig,
        max_psi=round(worst.psi, 5) if worst else float("nan"),
        worst_feature=worst.feature if worst else "",
        data_drift_detected=n_sig > 0,
        features=feats,
        model_drift=model_drift,
    )

    if save:
        PATHS.ensure()
        path = PATHS.reports / "drift_report.json"
        path.write_text(json.dumps(report.as_dict(), indent=2))
        pd.DataFrame([f.as_dict() for f in feats]).to_csv(
            PATHS.reports / "drift_features.csv", index=False)
        log.info("Wrote drift report -> %s", path)
    return report


def cloudwatch_metrics(report: DriftReport) -> list[dict]:
    """Shape for cloudwatch:PutMetricData (see deploy/aws/iam_policy.json)."""
    metrics = [
        {"MetricName": "DataDriftMaxPSI", "Value": float(report.max_psi), "Unit": "None"},
        {"MetricName": "DataDriftSignificantFeatures",
         "Value": float(report.n_significant), "Unit": "Count"},
        {"MetricName": "DataDriftModerateFeatures",
         "Value": float(report.n_moderate), "Unit": "Count"},
    ]
    if report.model_drift:
        metrics += [
            {"MetricName": "ModelRMSEvsBaseline",
             "Value": float(report.model_drift["rmse_vs_baseline"]), "Unit": "None"},
            {"MetricName": "ModelRMSEvsNaive",
             "Value": float(report.model_drift["rmse_vs_naive"]), "Unit": "None"},
        ]
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from .split import chronological_split

    feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
    splits = chronological_split(feats, save=False)
    rep = build_report(splits.train, splits.test)
    print(f"reference: {rep.n_reference} rows | current: {rep.n_current} rows "
          f"({rep.current_start} to {rep.current_end})")
    print(f"significant: {rep.n_significant}/{rep.n_features}  "
          f"moderate: {rep.n_moderate}  max PSI: {rep.max_psi:.3f} "
          f"({rep.worst_feature})")
    print()
    print(pd.DataFrame([f.as_dict() for f in rep.features])
          .head(12)[["feature", "psi", "ks_pvalue", "mean_shift_sigma", "severity"]]
          .to_string(index=False))
