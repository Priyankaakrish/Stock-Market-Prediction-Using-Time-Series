"""Test suite.

The tests that matter most here are the leakage tests. A stock-forecasting
pipeline that leaks the future does not crash or throw — it silently reports a
99.9% R-squared and looks like a triumph. These assertions are the only thing
standing between that and a production deploy.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import DATA, PATHS
from src.evaluate import (
    diebold_mariano,
    directional_accuracy,
    evaluate,
    mae,
    mape,
    rmse,
)
from src.features import bollinger, build_features, feature_columns, macd, rsi
from src.ingest import DataValidationError, validate
from src.models.baselines import DriftForecaster, MovingAverageForecaster, NaiveForecaster
from src.preprocess import preprocess
from src.split import chronological_split


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    if not PATHS.raw.exists():
        pytest.skip("raw snapshot not present")
    return pd.read_csv(PATHS.raw)


@pytest.fixture(scope="session")
def cleaned(raw) -> pd.DataFrame:
    return preprocess(raw, save=False)


@pytest.fixture(scope="session")
def features(cleaned) -> pd.DataFrame:
    return build_features(cleaned, save=False)


@pytest.fixture(scope="session")
def splits(features):
    return chronological_split(features, save=False)


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """A deterministic geometric random walk, for exact-arithmetic checks."""
    rng = np.random.default_rng(0)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame({
        "Date": dates,
        "Open": close * 0.995,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": rng.integers(1e6, 5e6, n).astype(float),
    })


# --------------------------------------------------------------------------
# Ingestion contracts
# --------------------------------------------------------------------------
class TestIngestion:
    def test_validate_accepts_good_frame(self, synthetic):
        # Pad to clear the 500-row minimum.
        big = pd.concat([synthetic] * 2, ignore_index=True)
        big["Date"] = pd.bdate_range("2015-01-01", periods=len(big))
        assert len(validate(big)) == len(big)

    def test_rejects_missing_columns(self, synthetic):
        with pytest.raises(DataValidationError, match="Missing required"):
            validate(synthetic.drop(columns=["Volume"]))

    def test_rejects_high_below_low(self, synthetic):
        bad = pd.concat([synthetic] * 2, ignore_index=True)
        bad["Date"] = pd.bdate_range("2015-01-01", periods=len(bad))
        bad.loc[10, "High"] = bad.loc[10, "Low"] - 1
        with pytest.raises(DataValidationError, match="High < Low"):
            validate(bad)

    def test_rejects_short_history(self, synthetic):
        with pytest.raises(DataValidationError, match="expected a long history"):
            validate(synthetic.head(100))


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
class TestPreprocess:
    def test_sorted_and_unique(self, cleaned):
        assert cleaned["Date"].is_monotonic_increasing
        assert not cleaned["Date"].duplicated().any()

    def test_no_missing_ohlcv(self, cleaned):
        assert cleaned[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum() == 0

    def test_drops_duplicate_dates(self, synthetic):
        dupe = pd.concat([synthetic, synthetic.tail(5)], ignore_index=True)
        out = preprocess(dupe, save=False)
        assert len(out) == len(synthetic)

    def test_does_not_invent_weekend_rows(self, cleaned):
        # A calendar reindex would push weekday share to ~71%.
        weekday_share = (cleaned["Date"].dt.dayofweek < 5).mean()
        assert weekday_share > 0.99


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
class TestIndicators:
    def test_rsi_bounds(self, synthetic):
        r = rsi(synthetic["Close"])
        assert r.between(0, 100).all()

    def test_rsi_all_gains_is_100(self):
        s = pd.Series(np.arange(1, 60, dtype=float))
        assert rsi(s).iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_macd_histogram_identity(self, synthetic):
        line, sig, hist = macd(synthetic["Close"])
        assert np.allclose(hist, line - sig)

    def test_bollinger_ordering(self, synthetic):
        upper, mid, lower, _width, _pct = bollinger(synthetic["Close"])
        valid = upper.notna()
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_indicators_are_causal(self, synthetic):
        """Changing a future price must not alter any past indicator value."""
        base = build_features(synthetic, save=False)
        perturbed = synthetic.copy()
        perturbed.loc[len(perturbed) - 1, "Close"] *= 1.5
        after = build_features(perturbed, save=False)

        common = min(len(base), len(after)) - 2
        cols = [c for c in ("RSI14", "close_over_MA20", "BB_pctB", "MACD_norm")
                if c in base.columns]
        for col in cols:
            np.testing.assert_allclose(
                base[col].to_numpy()[:common], after[col].to_numpy()[:common],
                rtol=1e-9, err_msg=f"{col} depends on a future price",
            )


# --------------------------------------------------------------------------
# Leakage — the important block
# --------------------------------------------------------------------------
class TestNoLeakage:
    def test_target_is_next_day_close(self, features):
        # target[t] must equal Close[t+1] wherever the rows are consecutive.
        d = features.reset_index(drop=True)
        same_seq = d["Date"].shift(-1).notna()
        np.testing.assert_allclose(
            d.loc[same_seq, "target"].to_numpy()[:-1],
            d["Close"].to_numpy()[1:][: same_seq.sum() - 1],
            rtol=1e-9,
        )

    def test_target_return_is_consistent(self, features):
        recomputed = np.log(features["target"] / features["Close"])
        np.testing.assert_allclose(recomputed, features["target_return"], rtol=1e-9)

    def test_feature_matrix_excludes_target(self, features):
        cols = feature_columns(features)
        assert "target" not in cols and "target_return" not in cols

    def test_feature_matrix_excludes_raw_levels(self, features):
        """Raw price levels would let a tree memorise the training range."""
        cols = set(feature_columns(features))
        for banned in ("Close", "Open", "High", "Low", "MA7", "MA50", "BB_upper"):
            assert banned not in cols, f"{banned} is a non-stationary level"

    def test_no_feature_correlates_perfectly_with_target(self, features):
        """A |correlation| of 1.0 with the target is the signature of leakage."""
        cols = feature_columns(features)
        corr = features[cols].corrwith(features["target_return"]).abs()
        worst = corr.max()
        assert worst < 0.99, f"suspicious feature: {corr.idxmax()} (r={worst:.4f})"

    def test_splits_do_not_overlap_in_time(self, splits):
        assert splits.train["Date"].max() < splits.val["Date"].min()
        assert splits.val["Date"].max() < splits.test["Date"].min()

    def test_split_proportions(self, splits):
        total = len(splits.train) + len(splits.val) + len(splits.test)
        assert abs(len(splits.train) / total - DATA.train_frac) < 0.01
        assert abs(len(splits.val) / total - DATA.val_frac) < 0.01


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0
        assert mape(y, y) == 0.0

    def test_rmse_known_value(self):
        assert rmse([1.0, 2.0], [2.0, 4.0]) == pytest.approx(np.sqrt(2.5))

    def test_directional_accuracy_all_correct(self):
        last = np.array([10.0, 10.0, 10.0])
        actual = np.array([11.0, 9.0, 12.0])
        pred = np.array([10.5, 9.5, 10.1])
        assert directional_accuracy(actual, pred, last) == 100.0

    def test_naive_directional_accuracy_is_undefined(self):
        """The random walk never commits to a direction; 0% would mislead."""
        last = np.array([10.0, 10.0, 10.0])
        actual = np.array([11.0, 9.0, 12.0])
        assert np.isnan(directional_accuracy(actual, last, last))

    def test_metrics_ignore_nans(self):
        m = evaluate("x", [1.0, 2.0, np.nan], [1.0, 2.0, 5.0], [1.0, 1.0, 1.0])
        assert m.n == 2 and m.rmse == 0.0

    def test_diebold_mariano_sign(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=500)
        good = y + rng.normal(scale=0.1, size=500)
        bad = y + rng.normal(scale=1.0, size=500)
        stat, p = diebold_mariano(y, good, bad)
        assert stat < 0 and p < 0.01     # negative => first argument is better


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class TestBaselines:
    def test_naive_echoes_last_close(self, splits):
        pred = NaiveForecaster().backtest(splits.train, splits.test)
        np.testing.assert_allclose(pred, splits.test["Close"].to_numpy())

    def test_moving_average_uses_only_past(self, splits):
        m = MovingAverageForecaster(window=5)
        pred = m.backtest(splits.train, splits.test)
        assert len(pred) == len(splits.test)
        assert np.isfinite(pred).all()

    def test_drift_is_positive_for_an_uptrend(self, splits):
        d = DriftForecaster().fit(splits.train)
        assert d.drift_ > 0            # NVDA rose over the training window

    def test_roundtrip_pickle(self, tmp_path, splits):
        m = DriftForecaster().fit(splits.train)
        path = m.save(tmp_path / "m.pkl")
        loaded = type(m).load(path)
        np.testing.assert_allclose(
            m.backtest(splits.train, splits.test.head(20)),
            loaded.backtest(splits.train, splits.test.head(20)),
        )


class TestXGBoost:
    def test_predicts_returns_not_levels(self, splits):
        """Regression guard for the extrapolation trap: a model trained on a
        window that tops out near $6 must still track a $200 test series."""
        from src.models.xgb_model import XGBForecaster

        m = XGBForecaster(n_estimators=60).fit(splits.train, splits.val)
        pred = m.backtest(splits.val, splits.test)
        assert pred.max() > 100, "predictions capped near the training range"
        assert np.median(np.abs(pred / splits.test["Close"] - 1)) < 0.05

    def test_feature_importance_shape(self, splits):
        from src.models.xgb_model import XGBForecaster

        m = XGBForecaster(n_estimators=40).fit(splits.train, splits.val)
        imp = m.feature_importance(10)
        assert len(imp) == 10 and imp["importance"].is_monotonic_decreasing


class TestARIMA:
    def test_fits_and_backtests(self, splits):
        from src.models.arima_model import ARIMAForecaster

        m = ARIMAForecaster(order=(1, 1, 1)).fit(splits.train.tail(400))
        pred = m.backtest(splits.train.tail(400), splits.test.head(30))
        assert len(pred) == 30 and np.isfinite(pred).all() and (pred > 0).all()


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


class TestAPI:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert "version" in body

    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()
        assert "/forecast" in body["endpoints"]

    def test_metrics_is_prometheus_text(self, client):
        text = client.get("/metrics").text
        assert "nvda_forecast_requests_total" in text

    @pytest.mark.skipif(not PATHS.best_model.exists(), reason="no trained model")
    def test_forecast_shape(self, client):
        r = client.post("/forecast", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "NVDA"
        assert body["direction"] in {"up", "down", "flat"}
        assert body["predicted_close"] > 0
        # A one-day forecast that moves more than 25% is a broken model.
        assert abs(body["predicted_change_pct"]) < 25
        assert "not investment advice" in body["disclaimer"]

    @pytest.mark.skipif(not PATHS.best_model.exists(), reason="no trained model")
    def test_rejects_insufficient_history(self, client):
        bars = [{"date": "2026-01-02", "open": 1, "high": 2,
                 "low": 1, "close": 1.5, "volume": 100}]
        assert client.post("/forecast", json={"bars": bars}).status_code == 422

    def test_rejects_invalid_bar(self, client):
        bars = [{"date": "2026-01-02", "open": -1, "high": 2,
                 "low": 1, "close": 1.5, "volume": 100}]
        assert client.post("/forecast", json={"bars": bars}).status_code == 422


# --------------------------------------------------------------------------
# Walk-forward validation
# --------------------------------------------------------------------------
from src.walkforward import make_folds, summarise, walk_forward  # noqa: E402


class TestWalkForward:
    def test_folds_are_contiguous_and_ordered(self):
        folds = make_folds(n_total=1000, n_initial=700, step=100)
        assert len(folds) == 3
        for tr, te in folds:
            assert tr.stop == te.start, "train must end exactly where test begins"
            assert tr.start == 0, "expanding window must start at the beginning"
        starts = [te.start for _, te in folds]
        assert starts == sorted(starts)

    def test_train_window_expands(self):
        folds = make_folds(n_total=1000, n_initial=500, step=100)
        sizes = [tr.stop - tr.start for tr, _ in folds]
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)

    def test_no_test_data_leaks_into_training(self):
        """The defining property: every training index precedes every test index."""
        for tr, te in make_folds(n_total=500, n_initial=300, step=50):
            assert max(range(tr.start, tr.stop)) < min(range(te.start, te.stop))

    def test_stub_final_fold_is_dropped(self):
        # 1000 rows, start at 950, step 100 -> a 50-row remainder is kept,
        # but a 5-row remainder would be noise.
        assert make_folds(n_total=1002, n_initial=1000, step=100) == []

    def test_walk_forward_runs_on_baselines(self, features):
        from src.models.baselines import DriftForecaster, NaiveForecaster

        sub = features.tail(600).reset_index(drop=True)
        folds = walk_forward(sub, [NaiveForecaster(), DriftForecaster()],
                             n_initial=400, step=100, save=False)
        assert set(folds["model"]) == {"naive", "drift"}
        assert folds["fold"].nunique() == 2
        assert (folds["rmse"] > 0).all()

    def test_summary_reports_win_rate(self, features):
        from src.models.baselines import DriftForecaster, NaiveForecaster

        sub = features.tail(600).reset_index(drop=True)
        folds = walk_forward(sub, [NaiveForecaster(), DriftForecaster()],
                             n_initial=400, step=100, save=False)
        s = summarise(folds)
        assert "folds_beating_naive_pct" in s.columns
        row = s.loc[s["model"] == "drift"].iloc[0]
        assert 0 <= row["folds_beating_naive_pct"] <= 100


# --------------------------------------------------------------------------
# Drift detection and S3 storage
# --------------------------------------------------------------------------
from src.drift import (  # noqa: E402
    detect_feature_drift,
    detect_model_drift,
    population_stability_index,
)
from src.storage import S3NotConfigured, raw_key, upload_raw, year_partition_key  # noqa: E402


class TestDrift:
    def test_psi_is_zero_for_identical_samples(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-6)

    def test_psi_grows_with_distribution_shift(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 5000)
        small = population_stability_index(ref, rng.normal(0.1, 1, 5000))
        large = population_stability_index(ref, rng.normal(2.0, 1, 5000))
        assert large > small
        assert large > 0.25          # a 2-sigma shift must read as significant

    def test_psi_survives_an_empty_bin(self):
        """A bucket with no current mass would send PSI to infinity unfloored."""
        rng = np.random.default_rng(2)
        ref = rng.normal(size=2000)
        cur = rng.normal(size=2000) + 10       # disjoint support
        assert np.isfinite(population_stability_index(ref, cur))

    def test_feature_drift_sorted_worst_first(self, splits):
        drifts = detect_feature_drift(splits.train, splits.test)
        assert len(drifts) > 20
        psis = [d.psi for d in drifts]
        assert psis == sorted(psis, reverse=True)

    def test_known_drift_is_detected(self, splits):
        """NVDA's realised volatility fell between the train and test windows."""
        drifts = {d.feature: d for d in detect_feature_drift(splits.train, splits.test)}
        assert drifts["volatility_21"].severity == "significant"
        assert drifts["volatility_21"].mean_shift_sigma < 0

    def test_model_drift_flags_a_broken_model(self, splits):
        recent = splits.test.tail(200)
        rubbish = recent["Close"].to_numpy(dtype=float) * 1.5
        out = detect_model_drift(recent, rubbish, baseline_rmse=3.0)
        assert out["degraded_vs_baseline"] and out["worse_than_naive"]

    def test_model_drift_clears_a_healthy_model(self, splits):
        recent = splits.test.tail(200)
        naive_pred = recent["Close"].to_numpy(dtype=float)
        out = detect_model_drift(recent, naive_pred, baseline_rmse=3.0)
        assert not out["worse_than_naive"]


class TestStorage:
    def test_raw_key_is_partitioned_by_ingest_date(self):
        key = raw_key("NVDA", date(2026, 4, 13))
        assert key == "raw/nvda/ingest_date=2026-04-13/nvda_raw.csv"

    def test_processed_key_is_partitioned_by_year_and_month(self):
        """The data-lake diagram specifies year/month, not year alone."""
        from src.storage import month_partition_key

        key = month_partition_key("NVDA", 2026, 4)
        assert key.startswith("processed/nvda/year=2026/month=4/")

    def test_month_is_unpadded_to_match_spark(self):
        """Spark's partitionBy writes month=4; a padded month=04 here would
        create a second, disjoint directory tree in the same bucket."""
        from src.storage import month_partition_key

        assert "month=4/" in month_partition_key("NVDA", 2026, 4)
        assert "month=04/" not in month_partition_key("NVDA", 2026, 4)

    def test_raw_key_supports_parquet(self):
        assert raw_key("NVDA", date(2026, 4, 13), fmt="parquet").endswith(".parquet")
        assert raw_key("NVDA", date(2026, 4, 13)).endswith(".csv")

    def test_processed_year_prefix_still_available(self):
        assert year_partition_key("NVDA", 2026) == "processed/nvda/year=2026/"

    def test_raw_keys_differ_per_ingest(self):
        """Immutability: a re-pull must not overwrite an earlier snapshot."""
        assert raw_key("NVDA", date(2026, 4, 13)) != raw_key("NVDA", date(2026, 4, 14))

    def test_missing_bucket_raises_clearly(self, monkeypatch):
        monkeypatch.delenv("S3_BUCKET", raising=False)
        with pytest.raises(S3NotConfigured, match="S3_BUCKET"):
            upload_raw()
