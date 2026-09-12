"""Serving-side monitoring.

The architecture's Monitoring box lists four things. Latency and error rate
were already counted in ``api/main.py``; this module adds the other two.

**Prediction distribution.** Latency and error rate tell you the service is
*up*. They say nothing about whether it has started emitting nonsense. A
forecaster that silently began predicting a 40% daily move would return 200s at
normal latency forever. So every prediction is recorded and summarised, and the
share of forecasts outside a plausible daily range is exported as its own
metric — that ratio is the alarm signal, not the mean.

**Drift.** ``src/drift.py`` computes it; nothing served it. The API now exposes
it at ``/drift`` so CloudWatch can scrape rather than requiring someone to run
a script.

The buffer is deliberately in-process and bounded. Prediction logging that
writes to a database on the request path adds a failure mode to inference in
exchange for telemetry, which is a bad trade; in production the ring buffer is
drained to CloudWatch by the agent on a timer.
"""
from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

log = logging.getLogger(__name__)

# A one-day move beyond this is possible but rare enough that a sustained rise
# in the breach rate means the model, not the market, has changed.
IMPLAUSIBLE_MOVE_PCT = 10.0
BUFFER_SIZE = 1000


@dataclass
class PredictionRecord:
    at: str
    as_of: str
    model: str
    last_close: float
    predicted_close: float
    change_pct: float


class PredictionMonitor:
    """Bounded, thread-safe ring buffer of recent predictions."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buf: deque[PredictionRecord] = deque(maxlen=maxlen)
        self._lock = Lock()

    def record(self, as_of, model: str, last_close: float,
               predicted_close: float, change_pct: float) -> None:
        rec = PredictionRecord(
            at=datetime.now(UTC).isoformat(),
            as_of=str(as_of), model=model,
            last_close=round(float(last_close), 6),
            predicted_close=round(float(predicted_close), 6),
            change_pct=round(float(change_pct), 6),
        )
        with self._lock:
            self._buf.append(rec)

    def snapshot(self) -> list[PredictionRecord]:
        with self._lock:
            return list(self._buf)

    def distribution(self) -> dict:
        """Summary of recent predicted moves — the shape monitoring cares about."""
        rows = self.snapshot()
        if not rows:
            return {"n": 0}

        changes = [r.change_pct for r in rows]
        implausible = [c for c in changes if abs(c) > IMPLAUSIBLE_MOVE_PCT]
        up = [c for c in changes if c > 0]

        out = {
            "n": len(changes),
            "mean_change_pct": round(statistics.fmean(changes), 6),
            "median_change_pct": round(statistics.median(changes), 6),
            "min_change_pct": round(min(changes), 6),
            "max_change_pct": round(max(changes), 6),
            # Share of forecasts that are directionally bullish. A model stuck
            # at 100% is the drift signature seen in this project's evaluation.
            "share_up_pct": round(100.0 * len(up) / len(changes), 4),
            "implausible_count": len(implausible),
            "implausible_rate_pct": round(100.0 * len(implausible) / len(changes), 4),
            "window_start": rows[0].at,
            "window_end": rows[-1].at,
        }
        if len(changes) > 1:
            out["stdev_change_pct"] = round(statistics.stdev(changes), 6)
        return out

    def prometheus_lines(self) -> list[str]:
        d = self.distribution()
        if not d.get("n"):
            return []
        return [
            "# HELP nvda_prediction_change_pct_mean Mean predicted daily move.",
            "# TYPE nvda_prediction_change_pct_mean gauge",
            f"nvda_prediction_change_pct_mean {d['mean_change_pct']}",
            "# HELP nvda_prediction_share_up_pct Share of bullish forecasts.",
            "# TYPE nvda_prediction_share_up_pct gauge",
            f"nvda_prediction_share_up_pct {d['share_up_pct']}",
            "# HELP nvda_prediction_implausible_rate_pct Forecasts beyond "
            f"±{IMPLAUSIBLE_MOVE_PCT}% — the alarm signal.",
            "# TYPE nvda_prediction_implausible_rate_pct gauge",
            f"nvda_prediction_implausible_rate_pct {d['implausible_rate_pct']}",
            "# HELP nvda_predictions_buffered Predictions held in the ring buffer.",
            "# TYPE nvda_predictions_buffered gauge",
            f"nvda_predictions_buffered {d['n']}",
        ]


MONITOR = PredictionMonitor()


def drift_summary(reference=None, current=None, max_features: int = 10) -> dict:
    """Data-drift summary for the ``/drift`` endpoint.

    Defaults to training vs the most recent window of the served dataset, which
    is the comparison that answers "is today's input like what we fitted on?"
    """
    import pandas as pd

    from src.config import PATHS
    from src.drift import detect_feature_drift
    from src.split import chronological_split

    if reference is None or current is None:
        if not PATHS.features.exists():
            return {"available": False,
                    "reason": "feature table not built; run python -m src.features"}
        feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
        splits = chronological_split(feats, save=False)
        reference = splits.train if reference is None else reference
        current = splits.test if current is None else current

    drifts = detect_feature_drift(reference, current)
    significant = [d for d in drifts if d.severity == "significant"]

    return {
        "available": True,
        "n_features": len(drifts),
        "n_significant": len(significant),
        "n_moderate": sum(d.severity == "moderate" for d in drifts),
        "max_psi": drifts[0].psi if drifts else None,
        "worst_feature": drifts[0].feature if drifts else None,
        "data_drift_detected": bool(significant),
        "top_features": [
            {"feature": d.feature, "psi": d.psi,
             "mean_shift_sigma": d.mean_shift_sigma, "severity": d.severity}
            for d in drifts[:max_features]
        ],
    }
