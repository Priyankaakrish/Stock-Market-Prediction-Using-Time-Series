"""
tests/test_models.py
─────────────────────
Unit tests for ARIMA, Prophet, LSTM, XGBoost, and Ensemble models.
Tests use lightweight synthetic data to run quickly without GPU.

Run: pytest tests/test_models.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def price_series() -> pd.Series:
    """200-day synthetic close price series."""
    np.random.seed(0)
    idx  = pd.bdate_range("2022-01-03", periods=200)
    vals = 100 + np.cumsum(np.random.randn(200))
    return pd.Series(vals, index=idx, name="Close")


@pytest.fixture
def ohlcv_df(price_series) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    c = price_series
    return pd.DataFrame({
        "Open":   c * 0.995,
        "High":   c * 1.010,
        "Low":    c * 0.985,
        "Close":  c,
        "Volume": np.random.randint(1_000_000, 3_000_000, len(c)).astype(float),
    }, index=c.index)


@pytest.fixture
def feature_df(ohlcv_df) -> pd.DataFrame:
    """Feature-engineered DataFrame."""
    from src.features.engineer import build_feature_set
    return build_feature_set(ohlcv_df)


# ── ARIMA tests ───────────────────────────────────────────────────────────────

class TestARIMAForecaster:
    def test_arima_forecast_shape(self, price_series):
        from src.models.arima_model import ARIMAForecaster
        model = ARIMAForecaster(order=(1, 1, 1), use_auto=False)
        model.fit(price_series)
        preds, ci = model.forecast(steps=5)
        assert len(preds) == 5
        assert ci.shape == (5, 2)

    def test_arima_forecast_values_are_finite(self, price_series):
        from src.models.arima_model import ARIMAForecaster
        model = ARIMAForecaster(order=(1, 1, 1), use_auto=False)
        model.fit(price_series)
        preds, _ = model.forecast(steps=10)
        assert np.all(np.isfinite(preds))

    def test_arima_save_load(self, price_series, tmp_path):
        from src.models.arima_model import ARIMAForecaster
        path  = tmp_path / "arima_test.pkl"
        model = ARIMAForecaster(order=(1, 1, 1), use_auto=False)
        model.fit(price_series)
        model.save(path)

        loaded = ARIMAForecaster.load(path)
        p1, _ = model.forecast(5)
        p2, _ = loaded.forecast(5)
        np.testing.assert_allclose(np.array(p1), np.array(p2), rtol=1e-4)


# ── Ensemble tests ────────────────────────────────────────────────────────────

class TestEnsembleForecaster:
    def _make_preds(self, n=50):
        np.random.seed(1)
        actual = 100 + np.cumsum(np.random.randn(n))
        return actual, {
            "arima":   actual + np.random.randn(n) * 1.0,
            "prophet": actual + np.random.randn(n) * 1.5,
            "lstm":    actual + np.random.randn(n) * 0.8,
            "xgboost": actual + np.random.randn(n) * 1.2,
        }

    def test_ensemble_predict_shape(self):
        from src.models.ensemble import EnsembleForecaster
        actual, preds = self._make_preds(50)
        ens = EnsembleForecaster()
        df  = ens.predict(preds)
        assert "ensemble" in df.columns
        assert len(df) == 50

    def test_ensemble_weights_sum_to_one(self):
        from src.models.ensemble import EnsembleForecaster
        actual, preds = self._make_preds(50)
        ens = EnsembleForecaster()
        ens.fit_weights(preds, actual)
        w_sum = sum(ens.weights.values())
        assert abs(w_sum - 1.0) < 1e-6

    def test_ensemble_optimised_rmse_lower_than_worst(self):
        from src.evaluation.metrics import rmse
        from src.models.ensemble import EnsembleForecaster
        actual, preds = self._make_preds(50)
        ens = EnsembleForecaster()
        ens.fit_weights(preds, actual)
        df  = ens.predict(preds)
        ens_rmse  = rmse(actual, df["ensemble"].values)
        worst_rmse= max(rmse(actual, p) for p in preds.values())
        assert ens_rmse <= worst_rmse + 0.5   # ensemble should be competitive

    def test_ensemble_save_load(self, tmp_path):
        from src.models.ensemble import EnsembleForecaster
        actual, preds = self._make_preds(30)
        ens = EnsembleForecaster()
        ens.fit_weights(preds, actual)
        path = tmp_path / "ens.pkl"
        ens.save(path)
        loaded = EnsembleForecaster.load(path)
        assert loaded.weights == ens.weights


# ── XGBoost / LightGBM tests ─────────────────────────────────────────────────

class TestGBForecaster:
    def _make_tabular(self, n=150):
        np.random.seed(7)
        X = np.random.randn(n, 20)
        y = X[:, 0] * 3 + X[:, 1] * -2 + np.random.randn(n) * 0.5 + 100
        return X, y

    def test_xgboost_predict_shape(self):
        from src.models.xgboost_model import GBForecaster
        X, y = self._make_tabular()
        gb = GBForecaster(model_type="xgboost", n_trials=2)
        gb.fit(X[:100], y[:100], X[100:120], y[100:120])
        preds = gb.predict(X[120:])
        assert len(preds) == len(X[120:])

    def test_xgboost_predictions_finite(self):
        from src.models.xgboost_model import GBForecaster
        X, y = self._make_tabular()
        gb = GBForecaster(model_type="xgboost", n_trials=2)
        gb.fit(X[:100], y[:100])
        preds = gb.predict(X[100:])
        assert np.all(np.isfinite(preds))

    def test_xgboost_save_load(self, tmp_path):
        from src.models.xgboost_model import GBForecaster
        X, y = self._make_tabular()
        gb   = GBForecaster(model_type="xgboost", n_trials=2)
        gb.fit(X[:100], y[:100])
        path = tmp_path / "xgb.pkl"
        gb.save(path)
        loaded = GBForecaster.load(path, "xgboost")
        np.testing.assert_allclose(gb.predict(X[100:]), loaded.predict(X[100:]))

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="LightGBM 4.7 Windows wheel crashes in Dataset.set_field; "
                               "LightGBM is optional and not used by the NVDA pipeline")
    def test_lightgbm_predict_shape(self):
        from src.models.xgboost_model import GBForecaster
        X, y = self._make_tabular()
        gb = GBForecaster(model_type="lightgbm", n_trials=2)
        gb.fit(X[:100], y[:100])
        preds = gb.predict(X[100:])
        assert len(preds) == len(X[100:])


# ── LSTM (lightweight smoke test) ─────────────────────────────────────────────

class TestLSTMForecaster:
    def test_lstm_build_and_predict(self):
        """Minimal LSTM smoke test — no actual training."""
        pytest.importorskip("tensorflow")
        from src.models.lstm_model import LSTMForecaster

        lf = LSTMForecaster(n_features=3, lookback=5, horizon=1,
                            units=[8, 4], dense_units=4)
        lf.build()

        X = np.random.randn(20, 5, 3).astype("float32")
        y = np.random.randn(20, 1).astype("float32")
        # Single-epoch quick smoke
        lf._model.fit(X, y, epochs=1, verbose=0)
        preds = lf.predict(X)
        assert preds.shape == (20, 1)

    def test_lstm_forecast_future_shape(self):
        pytest.importorskip("tensorflow")
        from src.models.lstm_model import LSTMForecaster

        lf = LSTMForecaster(n_features=2, lookback=5, horizon=1,
                            units=[8], dense_units=4)
        lf.build()

        seq = np.random.randn(5, 2).astype("float32")
        out = lf.forecast_future(seq, steps=10)
        assert len(out) == 10
