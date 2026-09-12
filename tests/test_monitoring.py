"""Tests for serving-side monitoring."""
from __future__ import annotations

import pytest

from api.monitoring import IMPLAUSIBLE_MOVE_PCT, PredictionMonitor


@pytest.fixture
def monitor():
    return PredictionMonitor(maxlen=100)


class TestPredictionDistribution:
    def test_empty_buffer_is_reported_not_crashed(self, monitor):
        assert monitor.distribution() == {"n": 0}
        assert monitor.prometheus_lines() == []

    def test_records_and_summarises(self, monitor):
        for pct in (0.1, -0.2, 0.3):
            monitor.record("2026-04-13", "drift", 100.0, 100.0 + pct, pct)
        d = monitor.distribution()
        assert d["n"] == 3
        assert d["min_change_pct"] == -0.2
        assert d["max_change_pct"] == 0.3
        assert d["share_up_pct"] == pytest.approx(200 / 3)

    def test_flags_implausible_moves(self, monitor):
        """A model emitting 40% daily moves still returns 200s at normal latency."""
        monitor.record("2026-04-13", "broken", 100.0, 140.0, 40.0)
        monitor.record("2026-04-14", "broken", 100.0, 100.1, 0.1)
        d = monitor.distribution()
        assert d["implausible_count"] == 1
        assert d["implausible_rate_pct"] == 50.0

    def test_plausible_moves_do_not_trip_the_alarm(self, monitor):
        monitor.record("2026-04-13", "drift", 100.0, 100.1, IMPLAUSIBLE_MOVE_PCT - 1)
        assert monitor.distribution()["implausible_rate_pct"] == 0.0

    def test_buffer_is_bounded(self):
        """Unbounded prediction logging is a memory leak on the request path."""
        m = PredictionMonitor(maxlen=10)
        for i in range(50):
            m.record("2026-04-13", "drift", 100.0, 100.1, 0.1 * i)
        assert m.distribution()["n"] == 10

    def test_all_bullish_is_visible(self, monitor):
        """The drift model's signature: every forecast up. Worth surfacing."""
        for _ in range(5):
            monitor.record("2026-04-13", "drift", 100.0, 100.1, 0.1)
        assert monitor.distribution()["share_up_pct"] == 100.0

    def test_prometheus_lines_are_well_formed(self, monitor):
        monitor.record("2026-04-13", "drift", 100.0, 100.1, 0.1)
        lines = monitor.prometheus_lines()
        names = [ln.split()[0] for ln in lines if not ln.startswith("#")]
        assert "nvda_prediction_implausible_rate_pct" in names
        assert all(len(ln.split()) == 2 for ln in lines if not ln.startswith("#"))


class TestMonitoringEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        pytest.importorskip("fastapi.testclient")
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as c:
            yield c

    def test_predictions_endpoint(self, client):
        assert "n" in client.get("/predictions").json()

    def test_drift_endpoint(self, client):
        body = client.get("/drift").json()
        assert body.get("available") is True
        assert body["n_significant"] >= 1
        assert body["worst_feature"] == "volatility_21"

    def test_root_advertises_the_new_endpoints(self, client):
        endpoints = client.get("/").json()["endpoints"]
        assert "/drift" in endpoints and "/predictions" in endpoints
