# S3 data-lake layout

```
s3://<bucket>/
├── raw/nvda/ingest_date=YYYY-MM-DD/nvda_raw.csv      immutable landing zone
│                                   nvda_raw.parquet  (optional, --fmt parquet)
├── processed/nvda/year=YYYY/month=M/*.parquet        year/month partitions
├── mlflow-artifacts/<experiment>/<run>/              models, plots, metrics
└── jobs/prepare_data_spark.py                        EMR job source
```

**Why year/month.** The feature table is queried by date range far more often
than any other way, so `year=/month=` lets Athena and Spark prune whole
directories: a query for Q1 2026 touches three files instead of 27 years of
history. Day partitioning would go too far — ~6,800 files of a few kilobytes,
where per-file overhead dominates the read. Month is the level where a partition
still holds ~21 trading days and the pruning is worth the metadata cost.

**Month is written unpadded** (`month=4`, not `month=04`) because that is what
Spark's `partitionBy("year", "month")` produces. A padded key on the pandas side
would create a second, disjoint directory tree in the same bucket, and the
pandas reader would silently miss every partition the EMR job wrote. A test
pins this.

**Why raw is immutable.** Vendors silently restate history — splits get
re-applied, bad ticks get corrected. Keeping every ingest under its own
`ingest_date` prefix means a model trained last quarter can be reproduced
exactly, rather than being re-trained against data that has quietly changed.

## Lifecycle policy

| Prefix              | Transition           | Expire     |
|---------------------|----------------------|------------|
| `raw/`              | Glacier IR at 90d    | never      |
| `processed/`        | Standard-IA at 30d   | 365d       |
| `mlflow-artifacts/` | Standard-IA at 60d   | never      |

Enable versioning and default SSE-KMS encryption on the bucket, and block all
public access at the account level.
