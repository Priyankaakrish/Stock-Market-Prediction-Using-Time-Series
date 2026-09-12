"""Tests for the model approval gate."""
from __future__ import annotations

from unittest.mock import patch

from src import promote


def _gate(rmse, mape, naive, incumbent=None):
    metrics = {"best_test_rmse": rmse, "best_test_mape": mape, "naive_test_rmse": naive}
    with patch.object(promote, "_metrics_for", return_value=metrics), \
         patch.object(promote, "current_production", return_value=incumbent):
        return promote.evaluate_gate()


class TestApprovalGate:
    def test_approves_a_healthy_model(self):
        assert _gate(rmse=2.99, mape=2.4, naive=3.0).approved

    def test_blocks_a_model_worse_than_the_random_walk(self):
        """The check this project exists to make."""
        result = _gate(rmse=3.5, mape=2.8, naive=3.0)
        assert not result.approved
        assert any("worse than the random walk" in r for r in result.reasons)

    def test_blocks_on_the_mape_ceiling(self):
        result = _gate(rmse=2.9, mape=27.2, naive=3.0)      # prophet-sized error
        assert not result.approved
        assert any("MAPE" in r and "exceeds" in r for r in result.reasons)

    def test_reports_when_no_baseline_is_available(self):
        result = _gate(rmse=2.99, mape=2.4, naive=None)
        assert any("cannot verify against a random walk" in r for r in result.reasons)

    def test_render_states_the_verdict(self):
        assert "REJECTED" in _gate(rmse=9.9, mape=8.0, naive=3.0).render()
        assert "APPROVED" in _gate(rmse=2.9, mape=2.4, naive=3.0).render()


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------
import pytest  # noqa: E402

from src.pipeline import PipelineError, run_pipeline  # noqa: E402


class TestOrchestrator:
    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="local, s3 or spark"):
            run_pipeline(mode="emr")

    def test_s3_mode_aborts_without_a_bucket(self, monkeypatch):
        """Silently skipping a stage the caller asked for is worse than failing."""
        monkeypatch.delenv("S3_BUCKET", raising=False)
        with pytest.raises(PipelineError, match="land raw"):
            run_pipeline(mode="s3", fast=True)

    def test_stage_result_carries_timing_and_outputs(self):
        from src.pipeline import StageResult

        r = StageResult("x", True, 1.5, "detail", ["out"])
        assert r.ok and r.seconds == 1.5 and r.outputs == ["out"]


class TestSplitStage:
    def test_split_stage_writes_the_three_tables(self):
        from src.config import PATHS
        from src.pipeline import _stage_split

        detail, outputs = _stage_split()
        assert "no leakage" in detail
        assert len(outputs) == 3
        for p in (PATHS.train, PATHS.val, PATHS.test):
            assert p.exists()

    def test_guard_rejects_a_random_split(self):
        """The failure mode it exists for: someone swaps in train_test_split.

        Shuffling the input file would not exercise this, because
        chronological_split sorts by date before splitting.
        """
        from unittest.mock import patch

        from src.config import PATHS
        from src.pipeline import _stage_split
        from src.split import Splits

        def random_split(df, save=True):
            d = df.sample(frac=1, random_state=0).reset_index(drop=True)
            n = len(d)
            return Splits(d.iloc[:int(n * .7)],
                          d.iloc[int(n * .7):int(n * .85)],
                          d.iloc[int(n * .85):])

        assert PATHS.features.exists()
        with patch("src.split.chronological_split", random_split):
            with pytest.raises(AssertionError):
                _stage_split()

    def test_split_reports_the_level_gap(self):
        """The 30x train-to-test price gap is why models learn returns."""
        from src.pipeline import _stage_split

        detail, _ = _stage_split()
        assert "x)" in detail and "->" in detail


class TestPipelineStages:
    def test_preprocess_stage_reports_no_calendar_reindex(self):
        """Forward-filling weekends would drop weekday share to ~71%."""
        from src.pipeline import _stage_preprocess

        detail, outputs = _stage_preprocess()
        assert "no calendar reindex" in detail
        assert len(outputs) == 1

    def test_feature_stage_reports_the_model_facing_split(self):
        """Raw price levels stay in the table but never reach a model."""
        from src.pipeline import _stage_features_pandas

        detail, _ = _stage_features_pandas()
        assert "model-facing" in detail
        assert "lags=" in detail and "volatility=" in detail

    def test_train_stage_names_the_time_series_models(self):
        """A champion reported without its baseline is the number to distrust."""
        from unittest.mock import patch

        import pandas as pd

        from src.pipeline import _stage_train

        table = pd.DataFrame({
            "model": ["drift", "xgboost", "naive", "arima"],
            "rmse": [2.99, 3.00, 3.01, 4.08],
        })
        with patch("src.train.run", return_value=table):
            detail, _ = _stage_train(fast=True)
        assert "time-series: arima/xgboost" in detail
        assert "baselines: naive/drift" in detail
        assert "vs naive" in detail


class TestServingVerification:
    def test_resolves_the_alias_and_predicts(self):
        from src.pipeline import _stage_verify_serving

        detail, outputs = _stage_verify_serving()
        assert "@Production ->" in detail
        assert outputs[0].startswith("models:/")

    def test_catches_a_broken_signature(self):
        """The int32 schema bug that shipped undetected for the project's life."""
        from unittest.mock import MagicMock, patch

        from mlflow.exceptions import MlflowException

        from src.pipeline import _stage_verify_serving

        broken = MagicMock()
        broken.predict.side_effect = MlflowException(
            "Incompatible input types for column dow. "
            "Can not safely convert int64 to int32.")
        with patch("mlflow.pyfunc.load_model", return_value=broken):
            with pytest.raises(MlflowException, match="int64 to int32"):
                _stage_verify_serving()

    def test_refuses_an_implausible_forecast(self):
        from unittest.mock import MagicMock, patch

        import pandas as pd

        from src.pipeline import PipelineError, _stage_verify_serving

        m = MagicMock()
        m.predict.return_value = pd.DataFrame(
            {"predicted_close": [280.0], "last_close": [188.63]})
        with patch("mlflow.pyfunc.load_model", return_value=m):
            with pytest.raises(PipelineError, match="daily move"):
                _stage_verify_serving()


class TestNoDuplicateWork:
    def test_train_reuses_the_orchestrator_tables(self):
        """The orchestrator times ingest/preprocess/features/split as separate
        stages, then calls train.run(). Without reuse, run() rebuilt all four
        silently — the pipeline did everything twice."""
        from unittest.mock import patch

        from src.train import build_dataset

        with patch("src.train.ingest") as ing, \
             patch("src.train.preprocess") as pre, \
             patch("src.train.build_features") as feat:
            feats, splits = build_dataset(reuse=True)

        ing.assert_not_called()
        pre.assert_not_called()
        feat.assert_not_called()
        assert len(feats) > 6000
        assert splits.train["Date"].max() < splits.test["Date"].min()

    def test_reuse_falls_back_when_no_table_exists(self):
        """A missing feature table must rebuild, not raise.

        PATHS is a frozen dataclass, so the absence is simulated by patching
        Path.exists rather than reassigning the path.
        """
        from unittest.mock import patch

        from src import train

        with patch("pathlib.Path.exists", return_value=False), \
             patch("src.train.ingest") as ing:
            ing.side_effect = RuntimeError("rebuild path taken")
            with pytest.raises(RuntimeError, match="rebuild path taken"):
                train.build_dataset(reuse=True)
