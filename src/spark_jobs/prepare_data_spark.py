"""PySpark feature-engineering job for AWS EMR.

The pandas path in ``src/features.py`` handles a single ticker's 27-year daily
history — about 6,800 rows, which fits comfortably in memory on a laptop. This
module is the same logic expressed in Spark for the case the architecture
diagram is really sizing for: a universe of tickers, or intraday bars, where
the feature table runs to hundreds of millions of rows.

The two implementations are kept deliberately parallel. ``--validate-against``
reads the pandas output and asserts the columns agree to within floating-point
tolerance, which is the only reliable way to stop the two paths drifting apart.

Run locally:
    spark-submit src/spark_jobs/prepare_data_spark.py \
        --input data/raw/nvda_raw.csv --output data/processed/spark

Run on EMR:
    spark-submit --deploy-mode cluster \
        s3://<bucket>/jobs/prepare_data_spark.py \
        --input  s3://<bucket>/raw/nvda/ \
        --output s3://<bucket>/processed/nvda/ \
        --partition-by year-month
"""
from __future__ import annotations

import argparse
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# Explicit schema: inferSchema triggers a second full pass over the data and
# will happily type a column as string because of one malformed row.
# Longest look-back in the feature set (the 60-day dollar-volume z-score in the
# pandas implementation). Rows before this have incomplete windows.
WARMUP_ROWS = 59

RAW_SCHEMA = StructType([
    StructField("Date", DateType(), False),
    StructField("Adj Close", DoubleType(), True),
    StructField("Close", DoubleType(), True),
    StructField("High", DoubleType(), True),
    StructField("Low", DoubleType(), True),
    StructField("Open", DoubleType(), True),
    StructField("Volume", LongType(), True),
])


def build_session(app_name: str = "nvda-feature-engineering",
                  s3: bool = False) -> SparkSession:
    """Build the session.

    ``s3=True`` pulls the hadoop-aws connector so ``s3a://`` paths resolve.
    PySpark installed from pip does not bundle it — on EMR the JARs are already
    on the classpath, which is why the job runs there unmodified and needs this
    only when writing to S3 from a laptop.

    The JAR versions must match the Hadoop version PySpark was built against,
    or you get a ``NoSuchMethodError`` at runtime rather than a clean failure.
    """
    builder = SparkSession.builder.appName(app_name)

    if s3:
        # hadoop-aws must track the Hadoop version PySpark was built against,
        # not the Spark version. Mismatch surfaces as NoSuchMethodError at
        # runtime rather than a clean dependency failure, so it is pinned.
        hadoop = os.getenv("HADOOP_AWS_VERSION", "3.3.4")
        builder = (
            builder
            .config("spark.jars.packages",
                    f"org.apache.hadoop:hadoop-aws:{hadoop},"
                    f"com.amazonaws:aws-java-sdk-bundle:1.12.262")
            .config("spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem")
            # Credentials come from the default chain: env vars, ~/.aws, or the
            # EC2 instance profile. Never hard-code them into the job.
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
            .config("spark.hadoop.fs.s3a.endpoint",
                    f"s3.{os.getenv('AWS_REGION', 'ap-south-1')}.amazonaws.com")
        )

    return (
        builder
        # Adaptive execution coalesces the many tiny partitions a windowed
        # job produces; without it EMR writes thousands of small Parquet files.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def clean(df):
    """Stage 3 equivalent: dedupe, drop bad rows, repair zero volume."""
    w_dedupe = Window.partitionBy("ticker", "Date").orderBy(F.col("Close").desc())
    df = (
        df.withColumn("_rn", F.row_number().over(w_dedupe))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    df = df.filter(
        (F.col("Close") > 0) & (F.col("Open") > 0)
        & (F.col("High") >= F.col("Low")) & F.col("Date").isNotNull()
    )

    w_ff = Window.partitionBy("ticker").orderBy("Date").rowsBetween(
        Window.unboundedPreceding, Window.currentRow)
    return df.withColumn(
        "Volume",
        F.when(F.col("Volume") > 0, F.col("Volume").cast("double")).otherwise(None),
    ).withColumn("Volume", F.last("Volume", ignorenulls=True).over(w_ff))


def add_features(df, horizon: int = 1):
    """Stage 4 equivalent. Every window is unbounded-preceding-to-current-row,
    which is what makes the transformation causal."""
    w = Window.partitionBy("ticker").orderBy("Date")

    def trailing(n: int):
        return w.rowsBetween(-(n - 1), 0)

    close, vol = F.col("Close"), F.col("Volume")

    df = df.withColumn("prev_close", F.lag("Close", 1).over(w))
    df = df.withColumn("log_ret", F.log(close / F.col("prev_close")))
    df = df.withColumn("ret_1d", close / F.col("prev_close") - 1.0)

    for n in (5, 10, 21):
        df = df.withColumn(f"ret_{n}d", close / F.lag("Close", n).over(w) - 1.0)

    for n in (7, 21):
        df = df.withColumn(f"volatility_{n}", F.stddev("log_ret").over(trailing(n)))

    # Moving averages, exposed as scale-free ratios (see features.py docstring).
    for n in (7, 20, 50):
        df = df.withColumn(f"MA{n}", F.avg("Close").over(trailing(n)))
        df = df.withColumn(f"close_over_MA{n}", close / F.col(f"MA{n}"))
        df = df.withColumn(f"Volume_MA{n}", F.avg("Volume").over(trailing(n)))
    df = df.withColumn("MA7_over_MA50", F.col("MA7") / F.col("MA50"))
    df = df.withColumn("MA20_over_MA50", F.col("MA20") / F.col("MA50"))
    df = df.withColumn("vol_over_MA20", vol / F.col("Volume_MA20"))

    # Bollinger bands on the 20-day window.
    sd20 = F.stddev("Close").over(trailing(20))
    upper = F.col("MA20") + 2.0 * sd20
    lower = F.col("MA20") - 2.0 * sd20
    df = df.withColumn("BB_width", (upper - lower) / F.col("MA20"))
    df = df.withColumn("BB_pctB", (close - lower) / (upper - lower))

    # RSI with a simple-moving-average smoothing. Wilder's exponential
    # smoothing is recursive and has no closed-form window expression in Spark
    # SQL; if exact parity with pandas is required, express it as a pandas UDF
    # over each ticker partition instead.
    delta = close - F.col("prev_close")
    df = df.withColumn("_gain", F.when(delta > 0, delta).otherwise(0.0))
    df = df.withColumn("_loss", F.when(delta < 0, -delta).otherwise(0.0))
    avg_gain = F.avg("_gain").over(trailing(14))
    avg_loss = F.avg("_loss").over(trailing(14))
    df = df.withColumn(
        "RSI14",
        F.when(avg_loss == 0, F.lit(100.0)).otherwise(
            100.0 - (100.0 / (1.0 + avg_gain / avg_loss))),
    ).drop("_gain", "_loss")

    # Candle shape and gaps.
    df = df.withColumn("hl_range", (F.col("High") - F.col("Low")) / close)
    df = df.withColumn("oc_change", (close - F.col("Open")) / F.col("Open"))
    df = df.withColumn("gap_open", F.col("Open") / F.col("prev_close") - 1.0)

    for lag in (1, 2, 3, 5, 7, 14, 30):
        df = df.withColumn(f"log_ret_lag_{lag}", F.lag("log_ret", lag).over(w))
        df = df.withColumn(f"close_over_close_lag_{lag}",
                           close / F.lag("Close", lag).over(w))

    df = df.withColumn("dow", F.dayofweek("Date") - 1)
    df = df.withColumn("month", F.month("Date"))
    df = df.withColumn("year", F.year("Date"))

    # The only forward-looking columns in the job.
    df = df.withColumn("target", F.lead("Close", horizon).over(w))
    df = df.withColumn("target_return", F.log(F.col("target") / close))

    # Spark's rowsBetween has no min_periods: a "50-day average" over the first
    # three rows is silently computed from three rows. pandas' rolling(50)
    # returns NaN there. Without this guard the two implementations disagree on
    # ~60 rows and every early statistic is quietly wrong.
    warmup = max(WARMUP_ROWS, 1)
    df = df.withColumn("_rn", F.row_number().over(Window.partitionBy("ticker").orderBy("Date")))
    df = df.filter(F.col("_rn") > warmup).drop("_rn")

    return df.filter(F.col("target").isNotNull() & F.col("MA50").isNotNull())


def validate(df) -> dict:
    """Data-quality checks emitted as job metrics (ship these to CloudWatch)."""
    total = df.count()
    stats = df.select(
        F.sum(F.when(F.col("Close").isNull(), 1).otherwise(0)).alias("null_close"),
        F.sum(F.when(F.col("target").isNull(), 1).otherwise(0)).alias("null_target"),
        F.sum(F.when(F.abs(F.col("log_ret")) > 0.5, 1).otherwise(0)).alias("extreme_moves"),
        F.min("Date").alias("min_date"),
        F.max("Date").alias("max_date"),
    ).collect()[0].asDict()
    stats["rows"] = total
    stats["distinct_dates"] = df.select("Date").distinct().count()
    if stats["distinct_dates"] != total:
        raise ValueError("Duplicate dates survived the cleaning stage")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Spark feature engineering for NVDA")
    ap.add_argument("--input", required=True, help="CSV or Parquet path (local or s3://)")
    ap.add_argument("--output", required=True, help="Destination for the feature table")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--horizon", type=int, default=1)
    # year/month is what the data-lake diagram specifies: a query for one
    # quarter touches three directories instead of 27 years of history.
    ap.add_argument("--partition-by", default="year-month",
                    choices=["year-month", "year", "none"])
    ap.add_argument("--validate-against", default=None,
                    help="pandas feature CSV to cross-check against")
    args = ap.parse_args()

    # Only pull the S3 connector when a path actually needs it; otherwise the
    # job would download ~200 MB of JARs to write to a local directory.
    needs_s3 = args.input.startswith("s3") or args.output.startswith("s3")
    spark = build_session(s3=needs_s3)
    spark.sparkContext.setLogLevel("WARN")

    reader = spark.read
    if args.input.endswith(".csv") or "/raw/" in args.input:
        raw = reader.schema(RAW_SCHEMA).option("header", True).csv(args.input)
    else:
        raw = reader.parquet(args.input)

    if "ticker" not in raw.columns:
        raw = raw.withColumn("ticker", F.lit(args.ticker).cast(StringType()))

    features = add_features(clean(raw), horizon=args.horizon).cache()

    stats = validate(features)
    print("=" * 60)
    for k, v in stats.items():
        print(f"{k:>18}: {v}")
    print("=" * 60)

    writer = features.write.mode("overwrite")
    if args.partition_by == "year-month":
        writer = writer.partitionBy("year", "month")
    elif args.partition_by == "year":
        writer = writer.partitionBy("year")
    writer.parquet(args.output)
    print(f"Wrote feature table -> {args.output}")

    if args.validate_against:
        import pandas as pd

        ref = pd.read_csv(args.validate_against, parse_dates=["Date"])
        got = features.select("Date", "close_over_MA20", "log_ret", "target").toPandas()
        got["Date"] = pd.to_datetime(got["Date"])
        print(f"pandas rows: {len(ref)}   spark rows: {len(got)}")
        merged = ref.merge(got, on="Date", suffixes=("_pd", "_spark"))
        for col in ("close_over_MA20", "log_ret", "target"):
            diff = (merged[f"{col}_pd"] - merged[f"{col}_spark"]).abs().max()
            print(f"max |pandas - spark| for {col}: {diff:.3e}")
            assert diff < 1e-6, f"{col} diverged between implementations"
        assert len(ref) == len(got), (
            f"row-count mismatch: pandas {len(ref)} vs spark {len(got)}. "
            "Check WARMUP_ROWS against the longest look-back in features.py.")
        print("Parity check passed")

    spark.stop()


if __name__ == "__main__":
    main()
