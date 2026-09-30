"""
tests/test_forecasting.py
─────────────────────────
Tests for the NVDA-specific pieces: trend-safe outlier handling, scale-free
features, one-step alignment (no look-ahead) and recursive forecasting.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessor import remove_outliers_iqr
from src.evaluation.metrics import directional_accuracy
from src.features.engineer import get_model_feature_columns
from src.forecasting import (
    TARGET_RET, add_target, make_model_frame, make_windows,
    recursive_forecast, returns_to_next_prices,
)


@pytest.fixture
def growth_prices():
    """Exponential growth x1000 over 2,000 days — like NVIDIA's history."""
    idx = pd.bdate_range("2010-01-01", periods=2000)
    rng = np.random.default_rng(1)
    close = np.exp(np.linspace(0, np.log(1000), 2000) + rng.normal(0, 0.01, 2000))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": 1e6}, index=idx)


def test_outlier_removal_keeps_growth_trend(growth_prices):
    cleaned = remove_outliers_iqr(growth_prices)
    # a level-based IQR clamp would flatten the top of the series
    assert np.allclose(cleaned["Close"].values, growth_prices["Close"].values)


def test_outlier_removal_fixes_bad_tick(growth_prices):
    bad = growth_prices.copy()
    bad.iloc[1000, bad.columns.get_loc("Close")] *= 10      # 10x typo
    cleaned = remove_outliers_iqr(bad)
    assert cleaned["Close"].iloc[1000] == pytest.approx(bad["Close"].iloc[999])


def test_model_features_are_scale_free(growth_prices):
    frame = make_model_frame(growth_prices)
    cols = get_model_feature_columns(frame)
    early = frame[cols].iloc[300:500].abs().mean()
    late = frame[cols].iloc[-200:].abs().mean()
    # price went up ~100x between the windows; features must not
    ratio = (late + 1e-9) / (early + 1e-9)
    assert ratio.max() < 10, ratio.sort_values().tail()
    assert "Close" not in cols and "SMA_20" not in cols


def test_target_is_next_day_return(growth_prices):
    df = add_target(make_model_frame(growth_prices))
    c = df["Close"].values
    assert df[TARGET_RET].iloc[10] == pytest.approx(np.log(c[11] / c[10]))
    assert np.isnan(df[TARGET_RET].iloc[-1])


def test_prediction_is_indexed_by_target_day(growth_prices):
    frame = make_model_frame(growth_prices)
    preds = returns_to_next_prices(frame, np.zeros(len(frame)))
    # zero predicted return => prediction for day t equals close of day t-1
    assert preds.index[0] == frame.index[1]
    assert preds.iloc[0] == pytest.approx(frame["Close"].iloc[0])


def test_windows_end_on_target_row():
    feats = np.arange(10, dtype=float).reshape(-1, 1)
    target = np.arange(10, dtype=float) / 100
    X, y, ends = make_windows(feats, target, lookback=3)
    assert X.shape == (8, 3, 1)
    assert X[0, -1, 0] == 2 and ends[0] == 2 and y[0, 0] == pytest.approx(2.0)


def test_recursive_forecast_compounds(growth_prices):
    path = recursive_forecast(growth_prices, lambda f: 0.01, steps=5)
    last = growth_prices["Close"].iloc[-1]
    assert np.allclose(path, last * np.exp(0.01 * np.arange(1, 6)))


def test_directional_accuracy_against_previous_close():
    prev = np.array([10, 10, 10, 10.0])
    actual = np.array([11, 9, 11, 9.0])
    pred = np.array([10.5, 9.5, 9.5, 10.5])
    assert directional_accuracy(actual, pred, prev) == 50.0
