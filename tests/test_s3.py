"""S3 data-lake tests against a mocked bucket.

Provisioning a real bucket to discover that `upload_processed` writes to a key
ending in "/" is an expensive way to find a typo. moto stands in for S3 so the
first real upload is not also the first execution of this code.
"""
from __future__ import annotations

import pandas as pd
import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

BUCKET = "nvda-test-bucket"
REGION = "ap-south-1"


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", BUCKET)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    with moto.mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION})
        yield client


def _keys(client) -> list[str]:
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET}
        if token:
            kw["ContinuationToken"] = token
        page = client.list_objects_v2(**kw)
        out += [o["Key"] for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return out
        token = page["NextContinuationToken"]


class TestRawLanding:
    def test_uploads_csv(self, s3):
        from src.storage import upload_raw

        uri = upload_raw()
        assert uri.endswith(".csv")
        assert "ingest_date=" in uri

    def test_uploads_parquet(self, s3):
        from src.storage import upload_raw

        assert upload_raw(fmt="parquet").endswith(".parquet")

    def test_uploads_json(self, s3):
        """The data-lake diagram lists CSV/JSON for the raw zone."""
        from src.storage import upload_raw

        assert upload_raw(fmt="json").endswith(".json")

    def test_json_is_line_delimited(self, s3):
        """JSONL, not a single array: an array must be parsed whole before the
        first row is available, and cannot be split across Spark partitions."""

        from src.storage import raw_key, upload_raw

        upload_raw(fmt="json")
        body = s3.get_object(Bucket=BUCKET, Key=raw_key(fmt="json"))["Body"].read()
        text = body.decode()
        assert not text.lstrip().startswith("[")
        first = text.splitlines()[0]
        import json as _json
        row = _json.loads(first)          # each line parses on its own
        assert {"Date", "Open", "High", "Low", "Close", "Volume"} <= set(row)

    def test_json_round_trips_every_row(self, s3):
        import io

        from src.config import PATHS
        from src.storage import raw_key, upload_raw

        upload_raw(fmt="json")
        body = s3.get_object(Bucket=BUCKET, Key=raw_key(fmt="json"))["Body"].read()
        got = pd.read_json(io.BytesIO(body), lines=True)
        assert len(got) == len(pd.read_csv(PATHS.raw))

    def test_rejects_an_unknown_format(self, s3):
        from src.storage import upload_raw

        with pytest.raises(ValueError, match="csv"):
            upload_raw(fmt="avro")

    def test_round_trips(self, s3, tmp_path):
        from src.storage import download_raw, upload_raw

        upload_raw()
        dest = download_raw(dest=tmp_path / "rt.csv")
        assert len(pd.read_csv(dest)) > 6000

    def test_ingest_dates_are_the_reproducibility_index(self, s3):
        from datetime import date

        from src.storage import list_ingest_dates, upload_raw

        upload_raw(ingest_date=date(2026, 4, 13))
        upload_raw(ingest_date=date(2026, 4, 14))
        assert list_ingest_dates() == ["2026-04-13", "2026-04-14"]


class TestProcessedPartitions:
    def test_writes_year_month_partitions(self, s3):
        from src.storage import upload_processed

        uris = upload_processed()
        assert len(uris) > 300
        assert all("year=" in u and "month=" in u for u in uris)

    def test_no_zero_length_directory_objects(self, s3):
        """A key ending in "/" is not a file. `upload_processed` used a prefix
        helper as an object key and produced exactly that."""
        from src.storage import upload_processed

        upload_processed()
        assert [k for k in _keys(s3) if k.endswith("/")] == []

    def test_partition_layout_matches_spark(self, s3):
        """Spark writes month=4; a padded month=04 here would make two trees."""
        from src.storage import upload_processed

        upload_processed()
        months = {k.split("month=")[1].split("/")[0] for k in _keys(s3)
                  if "month=" in k}
        assert not any(m.startswith("0") and len(m) == 2 for m in months)

    def test_every_row_is_written_exactly_once(self, s3):
        from src.config import PATHS
        from src.storage import upload_processed

        upload_processed()
        expected = len(pd.read_csv(PATHS.features))
        total = 0
        for k in _keys(s3):
            if not k.endswith(".parquet") or "processed/" not in k:
                continue
            body = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()
            import io
            total += len(pd.read_parquet(io.BytesIO(body)))
        assert total == expected


class TestSyncAll:
    def test_lands_both_stages(self, s3):
        from src.storage import sync_all

        out = sync_all()
        assert out["raw"].startswith("s3://")
        assert len(out["processed"]) > 300


class TestSecretsManager:
    """load_secret() was written for the Security & Access box and never
    called by anything. These tests exercise it against a mocked Secrets
    Manager so the code path is at least proven before it is relied on."""

    @pytest.fixture
    def secrets(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
        with moto.mock_aws():
            yield boto3.client("secretsmanager", region_name=REGION)

    def test_reads_a_json_secret(self, secrets):
        import json

        from src.config import load_secret

        secrets.create_secret(Name="nvda/market-data",
                              SecretString=json.dumps({"api_key": "abc123"}))
        assert load_secret("nvda/market-data", key="api_key") == "abc123"
        assert load_secret("nvda/market-data") == {"api_key": "abc123"}

    def test_reads_a_plain_string_secret(self, secrets):
        from src.config import load_secret

        secrets.create_secret(Name="nvda/token", SecretString="plain-value")
        assert load_secret("nvda/token") == "plain-value"

    def test_missing_secret_returns_none_not_raises(self, secrets):
        """A missing optional secret must not take the API down at startup."""
        from src.config import load_secret

        assert load_secret("nvda/does-not-exist") is None
