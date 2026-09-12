"""Tests for drift detection and the S3 storage layer."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS
from src.drift import (
    detect_feature_drift,
    detect_model_drift,
    population_stability_index,
)
from src.preprocess import preprocess
from src.split import chronological_split
from src.storage import S3NotConfigured, raw_key, upload_raw, year_partition_key


@pytest.fixture(scope="module")
def splits():
    if not PATHS.features.exists():
        pytest.skip("feature table not built; run `python -m src.features`")
    feats = pd.read_csv(PATHS.features, parse_dates=["Date"])
    return chronological_split(feats, save=False)


class TestPSI:
    def test_zero_for_identical_samples(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_grows_with_distribution_shift(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 5000)
        small = population_stability_index(ref, rng.normal(0.1, 1, 5000))
        large = population_stability_index(ref, rng.normal(2.0, 1, 5000))
        assert large > small
        assert large > 0.25          # a 2-sigma shift must read as significant

    def test_survives_an_empty_bin(self):
        """Disjoint support: an unfloored empty bucket sends PSI to infinity."""
        rng = np.random.default_rng(2)
        ref = rng.normal(size=2000)
        cur = rng.normal(size=2000) + 10
        assert np.isfinite(population_stability_index(ref, cur))

    def test_returns_nan_on_too_little_data(self):
        assert np.isnan(population_stability_index([1.0, 2.0, 3.0], [1.0]))


class TestFeatureDrift:
    def test_sorted_worst_first(self, splits):
        drifts = detect_feature_drift(splits.train, splits.test)
        assert len(drifts) > 20
        psis = [d.psi for d in drifts]
        assert psis == sorted(psis, reverse=True)

    def test_known_drift_is_detected(self, splits):
        """NVDA's realised volatility fell between the train and test windows.

        Training spans the dot-com bust and 2008; the test window does not.
        """
        drifts = {d.feature: d for d in detect_feature_drift(splits.train, splits.test)}
        assert drifts["volatility_21"].severity == "significant"
        assert drifts["volatility_21"].mean_shift_sigma < 0

    def test_no_drift_against_itself(self, splits):
        drifts = detect_feature_drift(splits.test, splits.test)
        assert all(d.severity == "stable" for d in drifts)


class TestModelDrift:
    def test_flags_a_broken_model(self, splits):
        recent = splits.test.tail(200)
        rubbish = recent["Close"].to_numpy(dtype=float) * 1.5
        out = detect_model_drift(recent, rubbish, baseline_rmse=3.0)
        assert out["degraded_vs_baseline"]
        assert out["worse_than_naive"]

    def test_clears_a_healthy_model(self, splits):
        recent = splits.test.tail(200)
        naive_pred = recent["Close"].to_numpy(dtype=float)
        out = detect_model_drift(recent, naive_pred, baseline_rmse=3.0)
        assert not out["worse_than_naive"]

    def test_reports_the_naive_comparison(self, splits):
        """Dollar RMSE rises with the price level; only the naive ratio matters."""
        recent = splits.test.tail(200)
        out = detect_model_drift(recent, recent["Close"].to_numpy(dtype=float), 3.0)
        assert "naive_rmse_same_window" in out
        assert out["rmse_vs_naive"] == pytest.approx(1.0, abs=1e-9)


class TestStorage:
    def test_raw_key_is_partitioned_by_ingest_date(self):
        assert raw_key("NVDA", date(2026, 4, 13)) == \
            "raw/nvda/ingest_date=2026-04-13/nvda_raw.csv"

    def test_processed_key_is_partitioned_by_year(self):
        assert year_partition_key("NVDA", 2026).startswith("processed/nvda/year=2026/")

    def test_raw_keys_differ_per_ingest(self):
        """Immutability: a re-pull must not overwrite an earlier snapshot."""
        assert raw_key("NVDA", date(2026, 4, 13)) != raw_key("NVDA", date(2026, 4, 14))

    def test_missing_bucket_raises_clearly(self, monkeypatch):
        monkeypatch.delenv("S3_BUCKET", raising=False)
        with pytest.raises(S3NotConfigured, match="S3_BUCKET"):
            upload_raw()


class TestPreprocessStillClean:
    def test_no_calendar_reindex(self):
        """Forward-filling weekends would invent ~30% fake rows."""
        if not PATHS.raw.exists():
            pytest.skip("raw snapshot missing")
        cleaned = preprocess(pd.read_csv(PATHS.raw), save=False)
        assert (cleaned["Date"].dt.dayofweek < 5).mean() > 0.99
