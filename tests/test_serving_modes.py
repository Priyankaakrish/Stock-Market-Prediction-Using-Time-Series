"""The API must serve identically from the registry and from the local pickle.

`MODEL_URI=models:/nvda-forecaster@Production` is documented in .env.example as
the production serving mode, and it returned 500 on every request: the registry
hands back a PyFuncModel exposing only `predict`, while the handler called
`backtest`, which only the pickle has. Nothing caught it because every test ran
in pickle mode.
"""
from __future__ import annotations

import importlib

import pytest

from src.config import MLFLOW, PATHS


def _client(monkeypatch, uri: str | None):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    if uri:
        monkeypatch.setenv("MODEL_URI", uri)
    else:
        monkeypatch.delenv("MODEL_URI", raising=False)

    import api.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _registry_ready() -> bool:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW.tracking_uri)
        MlflowClient().get_model_version_by_alias(MLFLOW.registered_model, "Production")
        return True
    except Exception:
        return False


needs_registry = pytest.mark.skipif(
    not _registry_ready(), reason="no Production alias; run src.promote promote")


needs_pickle = pytest.mark.skipif(
    not PATHS.best_model.exists(),
    reason="no trained model; run `python -m src.train --fast`")


@needs_pickle
class TestPickleMode:
    def test_serves(self, monkeypatch):
        with _client(monkeypatch, None) as c:
            assert c.get("/health").json()["model_loaded"]
            body = c.post("/forecast", json={}).json()
            assert body["predicted_close"] > 0
            assert body["model"] != "unknown"


@needs_registry
class TestRegistryMode:
    URI = f"models:/{MLFLOW.registered_model}@Production"

    def test_forecast_does_not_500(self, monkeypatch):
        """The regression: PyFuncModel has no .backtest()."""
        with _client(monkeypatch, self.URI) as c:
            r = c.post("/forecast", json={})
            assert r.status_code == 200, r.text
            assert r.json()["predicted_close"] > 0

    def test_health_reports_the_model_name(self, monkeypatch):
        """The registry branch used to return before loading metadata."""
        with _client(monkeypatch, self.URI) as c:
            assert c.get("/health").json()["model_name"] is not None

    def test_both_modes_agree(self, monkeypatch):
        """Same champion, same inputs — the numbers must match exactly."""
        with _client(monkeypatch, None) as c:
            pickle_out = c.post("/forecast", json={}).json()
        with _client(monkeypatch, self.URI) as c:
            registry_out = c.post("/forecast", json={}).json()

        assert registry_out["predicted_close"] == pytest.approx(
            pickle_out["predicted_close"], abs=1e-6)
        assert registry_out["last_close"] == pytest.approx(
            pickle_out["last_close"], abs=1e-6)
        assert registry_out["model"] == pickle_out["model"]

    def test_falls_back_when_the_uri_is_bad(self, monkeypatch):
        """A broken MODEL_URI must degrade to the pickle, not take the API down."""
        if not PATHS.best_model.exists():
            pytest.skip("no local pickle to fall back to")
        with _client(monkeypatch, "models:/does-not-exist@Production") as c:
            assert c.get("/health").json()["model_loaded"]
            assert c.post("/forecast", json={}).status_code == 200
