"""S3 data-lake layer.

Implements the storage stages the architecture calls for: raw bars land under
an immutable, date-partitioned prefix; the processed feature table is written
as year-partitioned Parquet; MLflow artifacts share the same bucket.

Two decisions worth stating.

Raw can land as CSV, Parquet or line-delimited JSON. CSV is what the vendor
returns and the faithful record of what arrived; Parquet is what anything
downstream wants to read; JSON exists because the data-lake diagram lists
``CSV/JSON`` for the raw zone and some ingestion tooling only speaks JSON. It
is written as JSONL rather than a single array so it streams and splits.

**Raw is append-only, keyed by ingest date.** Vendors silently restate history
— splits get re-applied, bad ticks corrected. If ``raw/`` is overwritten in
place, a model trained last quarter can never be reproduced, because the data
it saw no longer exists. Every pull gets its own ``ingest_date=`` prefix.

**Processed is partitioned by year, not by day.** The feature table is queried
by date range far more than any other way, so year partitions let Athena and
Spark prune whole directories. Day partitioning would produce ~6,800 files of a
few kilobytes each, where per-file overhead dominates the read.

Credentials come from the EC2 instance profile (see ``deploy/aws/iam_policy.json``)
or the standard AWS credential chain. Nothing here reads or writes keys.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from .config import DATA, PATHS

log = logging.getLogger(__name__)

RAW_PREFIX = "raw"
PROCESSED_PREFIX = "processed"


class S3NotConfigured(RuntimeError):
    """Raised when an S3 operation is attempted without a bucket configured."""


def _bucket(bucket: str | None = None) -> str:
    b = bucket or os.getenv("S3_BUCKET")
    if not b:
        raise S3NotConfigured(
            "Set S3_BUCKET (see .env.example) or pass bucket= explicitly.")
    return b


def _client(region: str | None = None):
    try:
        import boto3
    except ImportError as err:                            # pragma: no cover
        raise S3NotConfigured(
            "boto3 is not installed. Add it to requirements.txt for the S3 path."
        ) from err
    return boto3.client("s3", region_name=region or os.getenv("AWS_REGION", "ap-south-1"))


# --------------------------------------------------------------------------
# Key layout
# --------------------------------------------------------------------------
def raw_key(ticker: str = DATA.ticker, ingest_date: date | None = None,
            fmt: str = "csv") -> str:
    d = ingest_date or date.today()
    return (f"{RAW_PREFIX}/{ticker.lower()}/ingest_date={d.isoformat()}/"
            f"{ticker.lower()}_raw.{fmt}")


def processed_prefix(ticker: str = DATA.ticker) -> str:
    return f"{PROCESSED_PREFIX}/{ticker.lower()}/"


def month_partition_key(ticker: str, year: int, month: int) -> str:
    """Hive-style year=/month= layout, as the data-lake diagram specifies.

    Unpadded month, because that is what Spark's ``partitionBy("year","month")``
    writes. A padded ``month=04`` here and an unpadded ``month=4`` there would
    produce two disjoint directory trees in the same bucket, and the pandas
    reader would silently miss every partition the EMR job wrote.
    """
    return f"{processed_prefix(ticker)}year={year}/month={month}/part-0000.parquet"


def year_partition_key(ticker: str, year: int) -> str:
    """Retained for the year-level prefix (used when pruning a whole year)."""
    return f"{processed_prefix(ticker)}year={year}/"


# --------------------------------------------------------------------------
# Upload / download
# --------------------------------------------------------------------------
def upload_raw(
    path: str | Path | None = None,
    bucket: str | None = None,
    ticker: str = DATA.ticker,
    ingest_date: date | None = None,
    fmt: str = "csv",
) -> str:
    """Land raw bars under an immutable ingest-date prefix. Returns the s3 URI.

    Formats: ``csv`` (as the vendor returned it), ``json`` (line-delimited, for
    consumers expecting a document stream), ``parquet`` (what anything
    downstream actually wants to read). CSV is the default because a raw
    landing zone that silently transforms its input is no longer a faithful
    record of what arrived.
    """
    # Validate the argument before touching the filesystem: a typo'd format
    # should say so, not report a missing file and send you looking in the
    # wrong place.
    if fmt not in ("csv", "json", "parquet"):
        raise ValueError(f"fmt must be 'csv', 'json' or 'parquet', got {fmt!r}")

    path = Path(path or PATHS.raw)
    if not path.exists():
        raise FileNotFoundError(f"No raw file at {path}; run `python -m src.ingest`")

    b, key = _bucket(bucket), raw_key(ticker, ingest_date, fmt)

    upload_path = path
    if fmt != "csv":
        upload_path = Path("/tmp") / f"{ticker.lower()}_raw.{fmt}"
        frame = pd.read_csv(path)
        if fmt == "parquet":
            frame.to_parquet(upload_path, index=False, compression="snappy")
        else:
            # Line-delimited: Spark and Athena both read JSONL natively, where
            # a single top-level array would have to be parsed whole before any
            # row could be processed.
            frame.to_json(upload_path, orient="records", lines=True,
                          date_format="iso")

    _client().upload_file(str(upload_path), b, key,
                          ExtraArgs={"ServerSideEncryption": "AES256"})
    if fmt != "csv":
        upload_path.unlink(missing_ok=True)

    uri = f"s3://{b}/{key}"
    log.info("Uploaded raw (%s) -> %s", fmt, uri)
    return uri


def upload_processed(
    df: pd.DataFrame | None = None,
    bucket: str | None = None,
    ticker: str = DATA.ticker,
    tmp_dir: str | Path = "/tmp",
) -> list[str]:
    """Write the feature table as year/month-partitioned Parquet.

    Returns one s3 URI per partition. The layout matches what Spark's
    ``partitionBy("year", "month")`` writes, so the EMR job and this path
    populate the same directory tree rather than two disjoint ones.
    """
    if df is None:
        df = pd.read_csv(PATHS.features, parse_dates=["Date"])
    if "Date" not in df.columns:
        raise ValueError("Feature frame must carry a Date column to partition on")

    b = _bucket(bucket)
    client = _client()
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(df["Date"])
    staged = df.assign(_year=dates.dt.year, _month=dates.dt.month)

    uris = []
    for (year, month), part in staged.groupby(["_year", "_month"]):
        local = tmp_dir / f"{ticker.lower()}_{year}_{month}.parquet"
        part.drop(columns=["_year", "_month"]).to_parquet(
            local, index=False, compression="snappy")
        key = month_partition_key(ticker, int(year), int(month))
        client.upload_file(str(local), b, key,
                           ExtraArgs={"ServerSideEncryption": "AES256"})
        local.unlink(missing_ok=True)
        uris.append(f"s3://{b}/{key}")

    log.info("Uploaded %d year/month partitions -> s3://%s/%s",
             len(uris), b, processed_prefix(ticker))
    return uris


def list_ingest_dates(bucket: str | None = None, ticker: str = DATA.ticker) -> list[str]:
    """Every raw snapshot ever landed, newest last. The reproducibility index."""
    b = _bucket(bucket)
    paginator = _client().get_paginator("list_objects_v2")
    prefix = f"{RAW_PREFIX}/{ticker.lower()}/"
    dates = set()
    for page in paginator.paginate(Bucket=b, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            token = cp["Prefix"].rstrip("/").split("ingest_date=")[-1]
            dates.add(token)
    return sorted(dates)


def download_raw(
    bucket: str | None = None,
    ticker: str = DATA.ticker,
    ingest_date: str | None = None,
    dest: str | Path | None = None,
) -> Path:
    """Fetch a specific snapshot — the mechanism for reproducing an old run."""
    b = _bucket(bucket)
    if ingest_date is None:
        available = list_ingest_dates(b, ticker)
        if not available:
            raise FileNotFoundError(f"No raw snapshots under s3://{b}/{RAW_PREFIX}/")
        ingest_date = available[-1]

    key = f"{RAW_PREFIX}/{ticker.lower()}/ingest_date={ingest_date}/{ticker.lower()}_raw.csv"
    dest = Path(dest or PATHS.raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(b, key, str(dest))
    log.info("Downloaded s3://%s/%s -> %s", b, key, dest)
    return dest


def read_processed(bucket: str | None = None, ticker: str = DATA.ticker,
                   years: list[int] | None = None,
                   months: list[int] | None = None) -> pd.DataFrame:
    """Read partitions back. Passing years/months is the partition pruning.

    ``read_processed(years=[2026], months=[1, 2, 3])`` touches three objects;
    omitting both reads 27 years. Requires ``s3fs`` — pandas delegates
    ``s3://`` paths to it, and without it the call fails with an import error
    rather than anything about S3.
    """
    b = _bucket(bucket)
    base = f"s3://{b}/{processed_prefix(ticker)}"
    if not years:
        return pd.read_parquet(base)

    prefixes = []
    for y in years:
        if months:
            prefixes += [f"{base}year={y}/month={m}/" for m in months]
        else:
            prefixes.append(f"{base}year={y}/")
    return pd.concat([pd.read_parquet(p) for p in prefixes], ignore_index=True)


MODELS_PREFIX = "models"


def model_key(ticker: str, version, filename: str = "best_model.pkl") -> str:
    """Versioned model artefacts, keyed by registry version.

    The diagram's Best Approved Model box calls for the production artefact to
    live in S3, versioned. Keying by the MLflow registry version rather than a
    timestamp means the object store and the registry cannot disagree about
    which pickle is v5 — and rollback becomes a matter of pointing at a
    different prefix, not re-running training.
    """
    return f"{MODELS_PREFIX}/{ticker.lower()}/version={version}/{filename}"


def upload_model(version, bucket: str | None = None, ticker: str = DATA.ticker,
                 include_metadata: bool = True) -> list[str]:
    """Publish the approved artefact and its metadata under a versioned prefix."""
    if not PATHS.best_model.exists():
        raise FileNotFoundError(
            f"No model at {PATHS.best_model}; run `python -m src.train` first")

    b, client = _bucket(bucket), _client()
    candidates = [PATHS.best_model]
    if include_metadata:
        candidates.append(PATHS.models / "best_model_meta.json")

    uris = []
    for local in candidates:
        if not local.exists():
            continue
        key = model_key(ticker, version, local.name)
        client.upload_file(str(local), b, key,
                           ExtraArgs={"ServerSideEncryption": "AES256"})
        uris.append(f"s3://{b}/{key}")

    log.info("Published model v%s (%d objects)", version, len(uris))
    return uris


def download_model(version, bucket: str | None = None, ticker: str = DATA.ticker,
                   dest=None) -> Path:
    """Pull a specific version back. This is the rollback path."""
    b = _bucket(bucket)
    dest = Path(dest or PATHS.best_model)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(b, model_key(ticker, version), str(dest))
    log.info("Restored model v%s -> %s", version, dest)
    return dest


def list_model_versions(bucket=None, ticker: str = DATA.ticker) -> list[str]:
    b = _bucket(bucket)
    paginator = _client().get_paginator("list_objects_v2")
    out = set()
    for page in paginator.paginate(Bucket=b,
                                   Prefix=f"{MODELS_PREFIX}/{ticker.lower()}/",
                                   Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            out.add(cp["Prefix"].rstrip("/").split("version=")[-1])
    return sorted(out, key=lambda v: int(v) if v.isdigit() else 0)


def sync_all(bucket: str | None = None, ticker: str = DATA.ticker) -> dict:
    """Land both stages in one call — what the training job would invoke."""
    return {
        "raw": upload_raw(bucket=bucket, ticker=ticker),
        "processed": upload_processed(bucket=bucket, ticker=ticker),
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="S3 data-lake operations")
    ap.add_argument("action",
                    choices=["upload", "list", "download", "keys",
                             "upload-model", "list-models"])
    ap.add_argument("--version", default=None, help="model registry version")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--ingest-date", default=None)
    args = ap.parse_args()

    if args.action == "keys":
        # Works with no bucket and no network: shows the layout that would be used.
        print("raw (csv):     ", raw_key())
        print("raw (parquet): ", raw_key(fmt="parquet"))
        print("processed:     ", month_partition_key(DATA.ticker, 2026, 4))
        print("model:         ", model_key(DATA.ticker, 5))
    elif args.action == "upload":
        print(sync_all(bucket=args.bucket))
    elif args.action == "upload-model":
        if not args.version:
            raise SystemExit("--version is required (see python -m src.promote list)")
        for u in upload_model(args.version, bucket=args.bucket):
            print(u)
    elif args.action == "list-models":
        for v in list_model_versions(args.bucket):
            print(f"v{v}")
    elif args.action == "list":
        for d in list_ingest_dates(args.bucket):
            print(d)
    else:
        print(download_raw(args.bucket, ingest_date=args.ingest_date))
