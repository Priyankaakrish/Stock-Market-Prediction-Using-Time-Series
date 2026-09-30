"""
tests/test_api.py
──────────────────
FastAPI endpoint tests using httpx + TestClient.
These are integration tests — they do NOT require trained models;
they check routing, schema validation, and error handling.

Run: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_registry():
    """
    A minimal model registry with stub models for API testing.
    Avoids loading real TF / Prophet / XGBoost models in CI.
    """
    mock_arima = MagicMock()
    mock_arima.forecast.return_value = (
        pd.Series(np.full(30, 400.0)),
        pd.DataFrame({"lower": np.full(30, 390.0), "upper": np.full(30, 410.0)}),
    )

    mock_xgb = MagicMock()
    mock_xgb.predict_multi_step.return_value = np.full(30, 402.0)
    mock_xgb.predict.side_effect = lambda X: np.full(len(X), 0.001)  # +0.1 %/day

    # Small synthetic price history for the recursive (XGBoost/LSTM) path
    from src.features.engineer import get_model_feature_columns
    from src.forecasting import make_model_frame
    idx   = pd.bdate_range("2025-01-01", periods=300)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 300)))
    ohlcv = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                          "Close": close, "Volume": 1e6}, index=idx)
    feature_cols = get_model_feature_columns(make_model_frame(ohlcv))

    mock_ens = MagicMock()
    mock_ens.predict.return_value = pd.DataFrame({
        "arima":    np.full(30, 400.0),
        "xgboost":  np.full(30, 402.0),
        "ensemble": np.full(30, 401.0),
        "lower":    np.full(30, 390.0),
        "upper":    np.full(30, 410.0),
    })

    return {
        "models": {
            "arima":   mock_arima,
            "xgboost": mock_xgb,
        },
        "ensemble":         mock_ens,
        "ohlcv":            ohlcv,
        "feature_cols":     feature_cols,
        "last_data_date":   str(idx[-1].date()),
        "last_date_series": idx[-1],
        "last_features":    np.random.randn(1, 50),
        "last_sequence":    np.random.randn(60, 50).astype("float32"),
        "scaler":           MagicMock(
            inverse_transform_col=lambda x, col: x
        ),
    }


@pytest.fixture
def client(mock_registry):
    """Create a FastAPI TestClient with a mocked registry."""
    from api.main import create_app
    app = create_app()
    app.state.model_registry = mock_registry

    # Override lifespan so we don't reload data/models
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop(app):
        yield

    app.router.lifespan_context = _noop
    return TestClient(app)


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_status_healthy(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"

    def test_health_lists_models(self, client):
        data = client.get("/api/v1/health").json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)


# ── Root endpoint ─────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_service_key(self, client):
        data = client.get("/").json()
        assert "service" in data


# ── Forecast endpoint ─────────────────────────────────────────────────────────

class TestForecastEndpoint:
    def test_forecast_default_returns_200(self, client):
        r = client.post("/api/v1/forecast", json={"steps": 5, "models": ["arima"]})
        assert r.status_code == 200

    def test_forecast_response_schema(self, client):
        r    = client.post("/api/v1/forecast",
                           json={"steps": 5, "models": ["arima"]})
        data = r.json()
        assert "ticker" in data
        assert "forecasts" in data
        assert "forecast_days" in data

    def test_forecast_steps_validated(self, client):
        """Steps > 252 should return 422 Unprocessable Entity."""
        r = client.post("/api/v1/forecast", json={"steps": 500, "models": ["arima"]})
        assert r.status_code == 422

    def test_forecast_invalid_model_rejected(self, client):
        r = client.post("/api/v1/forecast",
                        json={"steps": 5, "models": ["nonexistent_model"]})
        assert r.status_code == 422

    def test_forecast_returns_correct_step_count(self, client):
        r    = client.post("/api/v1/forecast",
                           json={"steps": 7, "models": ["arima"]})
        data = r.json()
        if "arima" in data.get("forecasts", {}):
            assert len(data["forecasts"]["arima"]) == 7


# ── Models list endpoint ──────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200

    def test_models_returns_list(self, client):
        data = client.get("/api/v1/models").json()
        assert isinstance(data, list)

    def test_models_contain_loaded_names(self, client):
        data  = client.get("/api/v1/models").json()
        names = [m["model"] for m in data]
        assert "arima" in names or "xgboost" in names


# ── XGBoost recursive path & single prediction ────────────────────────────────

class TestRecursivePaths:
    def test_xgboost_forecast_compounds_returns(self, client, mock_registry):
        r = client.post("/api/v1/forecast", json={"steps": 5, "models": ["xgboost"]})
        assert r.status_code == 200
        pts = [p["predicted"] for p in r.json()["forecasts"]["xgboost"]]
        last = float(mock_registry["ohlcv"]["Close"].iloc[-1])
        expected = last * np.exp(0.001 * np.arange(1, 6))
        assert np.allclose(pts, expected, rtol=1e-4)

    def test_predict_single(self, client):
        r = client.post("/api/v1/predict/single", json={"close": 100, "steps": 3})
        assert r.status_code == 200
        assert r.json()["predicted"] > 0
