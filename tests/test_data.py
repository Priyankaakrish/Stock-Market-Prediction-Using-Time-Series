"""
tests/test_data.py
───────────────────
Unit tests for data loading, cleaning, and feature engineering.
Run: pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Minimal synthetic OHLCV DataFrame for testing."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(300) * 2)
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, 300).astype(float),
    }, index=idx)


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_master_exists(self):
        """Master CSV must exist in data/raw/."""
        from config import DATA_FILES
        assert DATA_FILES["master"].exists(), \
            f"Master dataset not found at {DATA_FILES['master']}"

    def test_load_all_sources_returns_dataframe(self):
        """load_all_sources() should return a non-empty DataFrame."""
        from src.data.loader import load_all_sources
        df = load_all_sources()
        assert isinstance(df, pd.DataFrame), "Expected DataFrame"
        assert len(df) > 0, "DataFrame is empty"

    def test_load_all_sources_has_close_column(self):
        from src.data.loader import load_all_sources
        df = load_all_sources()
        assert "Close" in df.columns

    def test_date_index_is_datetime(self):
        from src.data.loader import load_all_sources
        df = load_all_sources()
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_duplicate_dates(self):
        from src.data.loader import load_all_sources
        df = load_all_sources()
        assert not df.index.duplicated().any(), "Duplicate dates found"

    def test_data_covers_expected_range(self):
        from src.data.loader import load_all_sources
        df = load_all_sources()
        assert df.index.min().year <= 2000, "Data should start no later than 2000"
        assert df.index.max().year >= 2024, "Data should extend to at least 2024"


# ── Preprocessor tests ────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_clean_removes_weekend_dates(self, sample_ohlcv):
        from src.data.preprocessor import clean
        df = clean(sample_ohlcv)
        assert (df.index.dayofweek < 5).all(), "Weekend dates found after clean()"

    def test_clean_no_nan_in_close(self, sample_ohlcv):
        from src.data.preprocessor import clean
        df = clean(sample_ohlcv)
        assert df["Close"].isna().sum() == 0

    def test_clean_adds_return_columns(self, sample_ohlcv):
        from src.data.preprocessor import clean
        df = clean(sample_ohlcv)
        assert "Return" in df.columns
        assert "Log_Return" in df.columns

    def test_temporal_split_sizes(self, sample_ohlcv):
        from src.data.preprocessor import temporal_split
        train, val, test = temporal_split(sample_ohlcv)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_ohlcv)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_temporal_split_chronological(self, sample_ohlcv):
        """Train must precede val which must precede test."""
        from src.data.preprocessor import temporal_split
        train, val, test = temporal_split(sample_ohlcv)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_build_sequences_shape(self, sample_ohlcv):
        from src.data.preprocessor import build_sequences
        arr = sample_ohlcv[["Close", "Volume"]].values.astype("float32")
        X, y = build_sequences(arr, lookback=10, horizon=1)
        assert X.shape[1] == 10
        assert X.shape[2] == 2
        assert len(X) == len(y)

    def test_data_scaler_inverse(self, sample_ohlcv):
        from src.data.preprocessor import DataScaler
        scaler = DataScaler("minmax")
        df_sc  = scaler.fit_transform(sample_ohlcv, ["Close"])
        inverted = scaler.inverse_transform_col(df_sc["Close"].values, "Close")
        np.testing.assert_allclose(inverted, sample_ohlcv["Close"].values, rtol=1e-4)


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_add_technical_indicators_columns(self, sample_ohlcv):
        from src.features.engineer import add_technical_indicators
        df = add_technical_indicators(sample_ohlcv)
        for col in ["RSI", "MACD", "BB_Upper", "BB_Lower", "ATR", "EMA_20"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rsi_range(self, sample_ohlcv):
        from src.features.engineer import add_technical_indicators
        df = add_technical_indicators(sample_ohlcv)
        rsi = df["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI out of [0, 100]"

    def test_lag_features(self, sample_ohlcv):
        from src.features.engineer import add_lag_features
        df = add_lag_features(sample_ohlcv, lag_days=[1, 5])
        assert "Close_lag_1" in df.columns
        assert "Close_lag_5" in df.columns

    def test_calendar_features(self, sample_ohlcv):
        from src.features.engineer import add_calendar_features
        df = add_calendar_features(sample_ohlcv)
        assert "DayOfWeek" in df.columns
        assert "Month_sin" in df.columns

    def test_build_feature_set_no_nan(self, sample_ohlcv):
        from src.features.engineer import build_feature_set
        df = build_feature_set(sample_ohlcv)
        assert df.isna().sum().sum() == 0, "NaN values found in feature set"

    def test_build_feature_set_increases_columns(self, sample_ohlcv):
        from src.features.engineer import build_feature_set
        df = build_feature_set(sample_ohlcv)
        assert len(df.columns) > 20, "Expected at least 20 feature columns"


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_zero_for_perfect_pred(self):
        from src.evaluation.metrics import mae
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_rmse_positive(self):
        from src.evaluation.metrics import rmse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        assert rmse(y_true, y_pred) > 0

    def test_directional_accuracy_perfect(self):
        from src.evaluation.metrics import directional_accuracy
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert directional_accuracy(y, y) == 100.0

    def test_r_squared_perfect(self):
        from src.evaluation.metrics import r_squared
        y = np.array([1.0, 2.0, 3.0])
        assert abs(r_squared(y, y) - 1.0) < 1e-10

    def test_compare_models_returns_dataframe(self):
        from src.evaluation.metrics import compare_models
        y    = np.random.randn(100) + 100
        preds = {
            "model_a": y + np.random.randn(100) * 0.5,
            "model_b": y + np.random.randn(100) * 1.0,
        }
        df = compare_models(y, preds)
        assert isinstance(df, pd.DataFrame)
        assert "RMSE" in df.columns
