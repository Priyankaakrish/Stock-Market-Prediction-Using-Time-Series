# 📈 NVIDIA Stock Price Forecasting

> End-to-end time-series ML pipeline for forecasting NVIDIA (NVDA) stock prices using ARIMA, Prophet, XGBoost, and LSTM — benchmarked against random-walk baselines, served via FastAPI, tracked with MLflow, and deployed with Docker Compose and an AWS S3 data lake.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20IAM-orange?logo=amazonaws)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)
![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Finding](#key-finding)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [AWS Deployment](#aws-deployment)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Configuration](#docker-configuration)
- [Results](#results)

---

## Overview

This project builds a production-grade stock price forecasting system for NVIDIA (ticker: `NVDA`). It trains four time-series and ML models plus three baselines, exposes predictions through a REST API, tracks all experiments using MLflow with a Model Registry, and is fully containerized with an AWS S3 data lake.

The system supports:
- Historical OHLCV ingestion with retries, validation and offline fallback
- 46-feature causal, scale-free feature engineering pipeline
- Multi-model training pipeline (ARIMA, Prophet, XGBoost, LSTM) plus naive / MA / drift baselines
- Walk-forward validation and Diebold-Mariano significance testing
- REST API for real-time forecasting with an empirical prediction interval
- MLflow UI for experiment tracking, model comparison and stage promotion
- Data drift and prediction-distribution monitoring
- Docker-based deployment behind Nginx, with an S3 data lake in ap-south-1

**Dataset:** 6,847 daily bars, 1999-01-22 → 2026-04-13. Zero missing values, zero duplicate dates.

---

## Key Finding

> **No model in this project beats a random walk to any statistically significant degree.**

On a 1,019-day held-out window the best model improves RMSE over "tomorrow's price equals today's price" by **0.08%**, with a Diebold-Mariano p-value of **0.62**. Walk-forward validation across 16 rolling quarters puts its win rate at 10/16 — a coin flip gives 8, binomial p = 0.454.

The apparent 53.8% directional accuracy is fully explained by the base rate of up-days in the test window (**53.78%**): the winning model predicts *up* every single day.

This is the correct result for daily equity closes, not a pipeline failure. Projects reporting sub-1% MAPE without a baseline row are usually measuring a lagged copy of the input — which is exactly what the `naive` row quantifies. The value here is the harness that makes it visible: baselines, a significance test, and a base-rate comparison reported next to every headline number.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER                             │
└────────────────────┬───────────────────────┬────────────────────────┘
                     │                       │
                     ▼                       ▼
              Port 80/443 (Nginx)    /mlflow/ (proxied)
                     │                       │
┌────────────────────▼───────────────────────▼────────────────────────┐
│                     Docker Host / AWS EC2                           │
│                       Ubuntu 24.04 LTS                              │
│                      ap-south-1 (Mumbai)                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Docker Network (nvda-net)                   │   │
│  │                                                              │   │
│  │  ┌────────────┐    ┌──────────────┐    ┌─────────────────┐   │   │
│  │  │   nginx    │───▶│   nvda-api   │    │   nvda-mlflow   │   │   │
│  │  │  80 / 443  │    │  FastAPI +   │    │  MLflow Server  │   │   │
│  │  │            │───▶│  Gunicorn    │───▶│  + Registry     │   │   │
│  │  │ rate-limit │    │    :8000     │    │      :5002      │   │   │
│  │  │ TLS term.  │    │  (unpublished)    │  (unpublished)  │   │   │
│  │  └────────────┘    └──────────────┘    └────────┬────────┘   │   │
│  │   ONLY published         │                      │            │   │
│  │      ports               │                      │            │   │
│  └──────────────────────────┼──────────────────────┼────────────┘   │
│                             │                      │                │
│                    models/best_model.pkl    EBS /mnt/mlflow         │
└─────────────────────────────┼──────────────────────┼────────────────┘
                              │                      │
                              ▼                      ▼
                    ┌──────────────────────────────────────┐
                    │              AWS S3                  │
                    │  raw/ · processed/ · mlflow-artifacts│
                    │  versioned · KMS · lifecycle rules   │
                    └──────────────────────────────────────┘

Security Group Inbound Rules:
  443  TCP  0.0.0.0/0   HTTPS
  80   TCP  0.0.0.0/0   redirects to 443
  8000 TCP  CLOSED      internal only
  5002 TCP  CLOSED      proxied via /mlflow
  22   TCP  CLOSED      deploys go over AWS SSM
```

Two deliberate differences from the common pattern: only Nginx publishes ports, so 8000 and 5002 are never exposed; and port 22 stays closed because deployments arrive over AWS SSM, meaning no SSH key material in CI.

### Data Flow

```
NVIDIA OHLCV (Yahoo Finance / cached snapshot)
      │  6,847 rows · 1999-01-22 → 2026-04-13
      ▼
① Data Ingestion (yfinance)
      │  retries w/ backoff · schema validation · offline CSV fallback
      ▼
② Land raw in S3 ─────────────▶ raw/nvda/ingest_date=YYYY-MM-DD/
      │                          immutable, CSV / JSON / Parquet
      ▼
③ Preprocessing
      │  dedupe · sort · forward-fill · zero-volume repair
      │  NO calendar reindex (weekends are not missing data)
      ▼
④ Feature Engineering ────────▶ 46 causal, scale-free features
      ├── MA ratios      close_over_MA7 / MA20 / MA50, MA7_over_MA50
      ├── Oscillators    RSI14, MACD_norm, MACD_signal, MACD_hist
      ├── Bollinger      BB_width, BB_pctB
      ├── Volatility     volatility_7 / _21, ATR14_norm, vol_ratio
      ├── Lags           log_ret_lag_{1,2,3,5,7,14,30}
      ├── Volume         vol_over_MA20, log_vol_change, dollar_vol_z
      └── Calendar       dow, month, is_month_end, is_quarter_end
      ▼
⑤ Publish processed ──────────▶ processed/nvda/year=YYYY/month=M/
      │                          325 Parquet partitions
      ▼
⑥ Chronological Split (70 / 15 / 15) — NO shuffling, no leakage
      │  train 4,750 │ val 1,018 │ test 1,019
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │  ARIMA   │ │ Prophet  │ │ XGBoost  │ │    LSTM    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
│  ┌────▼────────────▼────────────▼─────────────▼──────┐  │
│  │   BASELINES: naive · ma5 · drift                  │  │
│  │   (the row most projects omit)                    │  │
│  └────────────────────┬──────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────┘
                        ▼
⑦ Evaluation — RMSE · MAE · MAPE · Directional Accuracy
                 · Diebold-Mariano · walk-forward (16 folds)
                        ▼
⑧ MLflow Tracking (params · metrics · artifacts · 8 figures)
                        ▼
⑨ Model Registry — nvda-forecaster, Staging / Production aliases
                        ▼
⑩ Approval gate → best_model.pkl → FastAPI
```

---

## Features

- **Multi-model forecasting** — ARIMA, Prophet, XGBoost, LSTM in one unified harness with a single interface
- **Baselines that keep models honest** — naive (random walk), 5-day MA, and drift, scored identically
- **Statistical significance** — Diebold-Mariano with Newey-West variance and the Harvey small-sample correction
- **Walk-forward validation** — 16 expanding-window folds, each model refitted from scratch
- **Return-based targets** — models learn log returns, not price levels (see [Models](#models) for why this is decisive)
- **Leakage controls with tests** — 7 explicit assertions, because look-ahead bias never throws an exception
- **REST API** — FastAPI with Swagger UI and an empirical 80% prediction interval
- **Drift detection** — PSI and KS tests on every feature, served live at `/drift`
- **Prediction monitoring** — distribution of recent forecasts, with an implausible-move alarm signal
- **Experiment tracking** — MLflow with SQLite backend, Model Registry, and a three-gate approval step
- **AWS S3 data lake** — immutable raw zone, year/month Parquet partitions, lifecycle rules, least-privilege IAM
- **Containerized** — multi-stage Docker build, non-root user, healthchecks, Nginx reverse proxy

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI + Gunicorn + Uvicorn workers |
| ML Models | statsmodels (ARIMA), Prophet, XGBoost, PyTorch (LSTM) |
| Baselines | Naive / Moving Average / Drift |
| Statistics | scipy (Diebold-Mariano, PSI, KS), statsmodels (ACF) |
| Experiment Tracking | MLflow 3.x (SQLite backend + Model Registry) |
| Data Source | yfinance (Yahoo Finance) + cached CSV snapshot |
| Feature Engineering | pandas, numpy (hand-rolled indicators — no ta-lib) |
| Big Data | PySpark 4.x (local + AWS EMR) |
| Cloud Storage | AWS S3 (boto3, s3fs) |
| Visualization | matplotlib |
| Testing | pytest (108 tests), moto (mocked AWS), ruff |
| Containerization | Docker (multi-stage) + Docker Compose |
| Web Server | Nginx (reverse proxy, TLS, rate limiting) |
| CI/CD | GitHub Actions + AWS OIDC + ECR + SSM |
| Cloud | AWS S3, IAM, (EC2 / EBS / CloudWatch written) |
| Region | ap-south-1 (Mumbai) |

---

## Project Structure

```
nvda-forecasting/
│
├── api/
│   ├── main.py                      # FastAPI: /forecast /health /model /metrics /drift
│   ├── monitoring.py                # Prediction distribution + drift summary
│   └── schemas.py                   # Pydantic request/response models
│
├── src/
│   ├── config.py                    # Paths, hyperparameters, Secrets Manager loader
│   ├── ingest.py                    # yfinance fetch, retries, validation, fallback
│   ├── preprocess.py                # Dedupe, sort, ffill, zero-volume repair
│   ├── features.py                  # 46 causal scale-free features
│   ├── split.py                     # Chronological split + walk-forward windows
│   ├── walkforward.py               # Rolling-origin validation + fold summary
│   ├── evaluate.py                  # RMSE/MAE/MAPE/DA + Diebold-Mariano
│   ├── drift.py                     # PSI / KS data drift + model drift vs naive
│   ├── storage.py                   # S3 data lake: raw + year/month partitions
│   ├── promote.py                   # Approval gate + Production alias
│   ├── pipeline.py                  # Orchestrator: local | s3 | spark
│   ├── train.py                     # Training, MLflow, selection, registry
│   ├── plots.py                     # 8 evaluation figures
│   ├── mlflow_model.py              # pyfunc wrapper for registry deployment
│   ├── models/
│   │   ├── base.py                  # Shared Forecaster interface
│   │   ├── baselines.py             # Naive / MovingAverage / Drift
│   │   ├── arima_model.py           # ARIMA + AIC grid search
│   │   ├── prophet_model.py         # Prophet w/ quarterly refit backtest
│   │   ├── xgb_model.py             # XGBoost on log returns
│   │   └── lstm_model.py            # PyTorch LSTM on log returns
│   └── spark_jobs/
│       └── prepare_data_spark.py    # PySpark/EMR features + parity check
│
├── deploy/
│   ├── Dockerfile.api               # Multi-stage API image (non-root)
│   ├── Dockerfile.mlflow            # MLflow tracking server
│   ├── docker-compose.yml           # nvda-net orchestration
│   ├── nginx/nvda.conf              # Reverse proxy, TLS, rate limiting
│   └── aws/
│       ├── ec2_bootstrap.sh         # Docker + SSM + CloudWatch + EBS mount
│       ├── iam_policy.json          # Least-privilege instance profile
│       ├── cloudwatch_agent.json    # Metrics + log shipping
│       ├── cloudwatch_alarms.json   # 7 alarms + SNS topic
│       └── s3_layout.md             # Data-lake layout + lifecycle policy
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # ADF, ACF, fat tails, the level gap
│
├── .github/workflows/ci-cd.yml      # Test → gate → build → scan → deploy
├── tests/                           # 108 tests (leakage checks first)
├── data/raw/nvda_raw.csv            # Committed snapshot for offline CI
├── models/                          # arima/prophet/xgb/lstm + best_model.pkl
├── reports/
│   ├── EVALUATION.md                # Full evaluation write-up
│   ├── metrics.csv                  # Model comparison table
│   ├── walkforward_folds.csv        # Per-fold results
│   └── figures/                     # 8 PNGs
│
├── Makefile
├── requirements.txt
└── requirements-serve.txt           # Inference only (no torch/prophet)
```

---

## Models

### 0. Baselines — naive / ma5 / drift

The row most stock-prediction projects omit. `naive` is `ŷ(t+1) = y(t)` — the random walk, and under weak-form market efficiency the theoretically optimal point forecast. Everything else is scored against it.

- **Speed:** instant
- **Why it matters:** a model reporting 2% MAPE that does not beat this has learned nothing

### 1. ARIMA

Classical statistical model fitted on **`log(Close)`** with `d=1`, making it an ARMA on log returns — the stationary representation. Fitting on raw prices of a stock that grew 5,000x produces heteroskedastic residuals and unstable coefficients.

- **Params:** `order=(2,1,2)`, linear trend (a constant is annihilated by differencing); `grid_search_order()` selects by AIC
- **Backtest:** `append(refit=False)` — realised values appended after each test day, so every forecast is a true one-step-ahead with fixed coefficients
- **Speed:** ~45s

### 2. Facebook Prophet

Additive trend + seasonality decomposition. Included per the brief, and the result is informative: it shows what happens when a smooth trend model meets a near-martingale.

- **Params:** `changepoint_prior_scale=0.10`, weekly + yearly seasonality, `interval_width=0.80`
- **Backtest:** retrained every 63 trading days (~quarterly) and rolled forward — a realistic production cadence
- **Speed:** ~115s (16 refits)

### 3. XGBoost

Gradient-boosted trees on 46 engineered features. **The target is `log(Close_t+1 / Close_t)`, not the price.**

> ⚠️ **The single most consequential decision in the project.** The chronological split puts the training maximum at **$6.23** while the test window reaches **$207.04**. Trees predict the mean of a leaf, so output is bounded by the training target range. A price-level XGBoost would emit a flat line near $6 for the entire test period — RMSE around $70 — while appearing to have trained successfully. Predicting returns removes the level entirely. A test guards against anyone undoing this.

- **Params:** `n_estimators=600`, `max_depth=4`, `lr=0.02`, `subsample=0.8`, `min_child_weight=20`, early stopping on validation, then refit on train+val at the tuned tree count
- **Speed:** < 5s

### 4. LSTM (PyTorch)

Sequence model over a 30-day window of the same stationary features. Same target choice — a min-max scaler fitted on training *prices* has the mirror problem: every test price maps above 1.0, into a region the network never saw.

- **Architecture:** 2-layer LSTM (48 hidden) → Dropout → Dense(32) → Output
- **Params:** `lookback=30`, `dropout=0.2`, `lr=1e-3`, Huber loss, gradient clipping, early stopping (patience 6)
- **Speed:** ~20s on CPU

---

## API Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "drift",
  "trained_at": "2026-09-12T17:11:04+00:00",
  "data_rows": 6847,
  "data_last_date": "2026-04-13",
  "version": "1.0.0"
}
```

### `POST /forecast`

Empty body uses the server's cached history. Supply `bars` (≥ 120, ascending) to forecast from your own data.

**Request:**
```json
{ "bars": null }
```

**Response:**
```json
{
  "ticker": "NVDA",
  "model": "drift",
  "as_of": "2026-04-13",
  "last_close": 189.31,
  "predicted_close": 189.5097,
  "predicted_change": 0.1997,
  "predicted_change_pct": 0.1055,
  "direction": "up",
  "horizon_days": 1,
  "prediction_interval_80": [183.7502, 195.2692],
  "disclaimer": "Model output for research and educational use only..."
}
```

> The 80% interval is computed from realised 60-day volatility, **not** from the model. None of these point forecasters produces a calibrated predictive distribution, and presenting one as if it did would be dishonest.

### `GET /model`

Deployed model metadata and its held-out metrics, including the naive RMSE it should be judged against.

### `GET /predictions`

Distribution of recent forecasts. `implausible_rate_pct` is the alarm signal — latency says the service is up, this says whether it has started emitting nonsense.

```json
{"n": 1, "mean_change_pct": 0.10548, "share_up_pct": 100.0,
 "implausible_rate_pct": 0.0}
```

`share_up_pct: 100.0` is not a small-sample artifact — the deployed drift model is bullish on every call by construction.

### `GET /drift`

Live PSI / KS data drift against the training distribution.

### `GET /metrics`

Prometheus text format: request counts, error counts, latency, and prediction-distribution gauges. Restricted to private CIDRs in the Nginx config.

---

## Local Setup

### Prerequisites

- Python 3.12+, Git, (optional) Docker Desktop

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<yourusername>/nvda-forecasting.git
cd nvda-forecasting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# 4. Run the full offline pipeline
python -m src.pipeline --mode local --fast --walk-forward --promote

# 5. Run the test suite
python -m pytest tests -q       # 108 tests

# 6. Start MLflow UI locally
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
# Open: http://localhost:5002

# 7. Start FastAPI locally
python -m uvicorn api.main:app --port 8000
# Open: http://localhost:8000/docs
```

Expected pipeline output:

```
ingest                  ok    0.3s  6847 rows, 1999-01-22 to 2026-04-13
preprocess              ok    0.1s  6847 rows, 100.0% weekdays
feature engineering     ok    0.7s  6787 rows, 46/65 columns model-facing
train/val/test split    ok    0.6s  4750/1018/1019, $6.23 -> $207.04 (33x)
train + evaluate        ok  141.3s  best=drift 2.9977 vs naive 3.0000
walk-forward validation ok   24.1s  16 folds; drift 10/16 p=0.454
approval gate           ok    0.1s  approved
verify serving path     ok    0.2s  models:/nvda-forecaster@Production
8 stages, 170.0s total
```

---

## AWS Deployment

### Infrastructure

| Setting | Value |
|---|---|
| Region | ap-south-1 (Mumbai) |
| S3 bucket | versioned, SSE-S3, all public access blocked |
| IAM | dedicated user, scoped to three prefixes, no console access |
| Instance (planned) | t3.large, Ubuntu 24.04, 30 GB root + 20 GB EBS |
| Deploy channel | AWS SSM (no SSH) |

### Deployment status

Honest accounting, because "cloud-ready" usually means "never left the laptop".

| Component | State |
|---|---|
| S3 data lake — raw (CSV / JSON / Parquet) | **Live.** `raw/nvda/ingest_date=YYYY-MM-DD/`, immutable |
| S3 data lake — processed | **Live.** 325 `year=/month=` Parquet partitions, 16 MB |
| S3 — MLflow artifact store | **Live.** `mlflow-artifacts/models/…/python_model.pkl` |
| S3 — versioning + lifecycle | **Live.** 4 rules: Glacier IR, Standard-IA, multipart abort |
| IAM — least privilege | **Live.** Bucket-root `ls` correctly denied |
| Docker + Nginx stack | **Validated locally.** Both containers healthy, served via proxy |
| PySpark / EMR | Runs locally with exact pandas parity; S3 connector untested |
| Secrets Manager | Code written, tested against a mock, no secret stored |
| EC2 / EBS / CloudWatch / alarms / SGs / VPC | Written, never executed — needs a paid instance |

> The IAM denial is worth dwelling on. `aws s3 ls s3://bucket/ --recursive` fails with AccessDenied while `aws s3 ls s3://bucket/raw/` succeeds, because the policy grants `ListBucket` only under a prefix condition. Least privilege you can watch refuse something is least privilege that actually exists.

### Step 1 — Create the S3 bucket

AWS Console → S3 → Create bucket, region **ap-south-1**. Block all public access, enable **Versioning**, default encryption SSE-S3. Object Lock stays disabled — versioning gives the protection without the irreversibility.

### Step 2 — Create a least-privilege IAM user

IAM → Users → Create user (no console access) → Attach policies directly → Create policy → JSON:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3DataLakeObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::<bucket>/raw/*",
        "arn:aws:s3:::<bucket>/processed/*",
        "arn:aws:s3:::<bucket>/mlflow-artifacts/*"
      ]
    },
    {
      "Sid": "S3ListScopedToPrefixes",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::<bucket>",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["raw/*", "processed/*", "mlflow-artifacts/*"]
        }
      }
    }
  ]
}
```

Then Security credentials → Create access key → CLI → **download the CSV** (the secret is shown once).

### Step 3 — Configure credentials

```bash
aws configure
# Access Key ID     : 20 chars, starts AKIA
# Secret Access Key : 40 chars, mixed case
# Region            : ap-south-1
# Output            : json

aws sts get-caller-identity      # verify the ARN is the IAM user, not root
```

### Step 4 — Populate the data lake

```bash
export S3_BUCKET=<bucket>
export AWS_REGION=ap-south-1
python -m src.storage keys        # prints the layout, writes nothing
python -m src.storage upload      # raw + 325 partitions
```

Verify:

```bash
aws s3 ls s3://<bucket>/raw/ --recursive
aws s3 ls s3://<bucket>/processed/ --recursive --summarize | tail -3
# Total Objects: 325
```

### Step 5 — Point MLflow artifacts at S3

The artifact location is fixed when an experiment is created, so a new experiment name is required:

```bash
export MLFLOW_EXPERIMENT=nvda-price-forecasting-s3
export MLFLOW_ARTIFACT_ROOT=s3://<bucket>/mlflow-artifacts
python -m src.train --fast
aws s3 ls s3://<bucket>/mlflow-artifacts/ --recursive
```

### Step 6 — Apply lifecycle rules

S3 → bucket → Management → Create lifecycle rule, four of them:

| Rule | Prefix | Action |
|---|---|---|
| `raw-to-glacier-ir` | `raw/` | Glacier Instant Retrieval at 90 days. **No expiration.** |
| `processed-to-ia-then-expire` | `processed/` | Standard-IA at 30 days, expire at 365 |
| `mlflow-artifacts-to-ia` | `mlflow-artifacts/` | Standard-IA at 60 days |
| `abort-incomplete-multipart` | (whole bucket) | Abort incomplete uploads after 7 days |

`raw/` never expires — that is the reproducibility guarantee. Processed partitions are regenerable from raw in ninety seconds.

### Step 7 — Run the pipeline through S3

```bash
python -m src.pipeline --mode s3 --fast --promote
```

Adds `land raw -> S3` and `publish processed` as timed stages between ingest and training.

### Step 8 — EC2 (written, not executed)

`deploy/aws/ec2_bootstrap.sh` installs Docker, the SSM agent, the CloudWatch agent, formats and mounts the EBS volume at `/mnt/mlflow`, and configures log rotation. Deployment then runs:

```bash
cd /opt/nvda-forecasting
docker compose -f deploy/docker-compose.yml up -d --build
curl localhost/health
```

### Expand EBS Volume (if disk full)

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h
docker image prune -af      # old ECR images are the usual culprit
```

### After EC2 Reboot

`restart: unless-stopped` on every service means the stack returns on its own, and `nofail` in `/etc/fstab` keeps the EBS mount from blocking boot. With an Elastic IP assigned, the address does not change — no manual reattachment required.

> **Tip:** Assign an AWS Elastic IP to keep a permanent public address.

---

## MLflow Experiment Tracking

MLflow UI: `http://localhost:5002` locally, or `https://<host>/mlflow/` behind Nginx.

> **Note:** MLflow 3.x put the plain-file store into maintenance mode and raises on `file://` URIs. This project uses the **SQLite backend**, which is also what the EBS-backed server uses in production.

Each pipeline run creates a parent run with nested child runs per model.

**Parameters tracked:**
- Model type (arima/prophet/xgboost/lstm/naive/ma5/drift)
- All hyperparameters per model, plus the target representation (`log_return` vs `log_close`)
- Feature count, split boundaries, train/test date ranges

**Metrics tracked:**
- `val_rmse`, `val_mae`, `val_mape`, `val_directional_accuracy`
- `test_rmse`, `test_mae`, `test_mape`, `test_directional_accuracy`, `test_r2`
- `fit_seconds`, `test_up_day_base_rate`, `best_skill_vs_naive_pct`, `naive_test_rmse`

**Artifacts stored:**
- Trained model pickle per model
- 8 figures (splits, predictions, zoom, comparison, importance, residuals, LSTM curve, walk-forward)
- `metrics.csv`, `test_predictions.csv`, `feature_importance.csv`, split summary

**Tags:** `best_model`, `best_directional_model`, `beats_random_walk` — the caveat sits next to the result in the UI, not buried in a report.

### Experiment Structure

```
nvda-price-forecasting/
└── pipeline-20260912-171104/            (parent run)
    ├── naive     → test_rmse=3.000  mape=2.39%  DA=n/a
    ├── ma5       → test_rmse=4.230  mape=3.62%  DA=49.41%
    ├── drift     → test_rmse=2.998  mape=2.38%  DA=53.83%   ★ champion
    ├── arima     → test_rmse=4.083  mape=3.42%  DA=49.02%
    ├── prophet   → test_rmse=26.20  mape=27.22% DA=50.10%
    ├── xgboost   → test_rmse=2.999  mape=2.38%  DA=52.16%
    └── lstm      → test_rmse=2.998  mape=2.38%  DA=53.54%
```

### Model Registry

The champion is logged as a **pyfunc model**, not a bare pickle — a bare pickle records the file but cannot be versioned or staged. Registered as `nvda-forecaster` with a signature, with Staging and Production aliases:

```bash
python -m src.promote list       # every version and its alias
python -m src.promote check      # evaluate the newest against the gate
python -m src.promote promote    # set the Production alias
```

Three gates: not materially worse than the incumbent, MAPE under 5%, and — the one this project exists to make — **not worse than a random walk**.

```bash
export MODEL_URI=models:/nvda-forecaster@Production
```

---

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc g++
WORKDIR /wheels
COPY requirements-serve.txt .
RUN pip wheel --wheel-dir /wheels -r requirements-serve.txt

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements-serve.txt .
RUN pip install --no-index --find-links=/wheels -r requirements-serve.txt && rm -rf /wheels
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", "--bind", "0.0.0.0:8000", "--timeout", "60"]
```

> The serving image excludes torch and prophet. Shipping a 2 GB CUDA stack to serve a model that is a few hundred bytes of coefficients is waste.

### docker-compose.yml

```yaml
services:
  nvda-api:
    build: {context: .., dockerfile: deploy/Dockerfile.api}
    expose: ["8000"]                    # NOT published
    environment:
      MLFLOW_TRACKING_URI: http://nvda-mlflow:5002
      MODEL_URI: ${MODEL_URI:-}
    depends_on:
      nvda-mlflow: {condition: service_healthy}
    networks: [nvda-net]
    restart: unless-stopped

  nvda-mlflow:
    build: {context: .., dockerfile: deploy/Dockerfile.mlflow}
    expose: ["5002"]                    # NOT published
    volumes: [mlflow-data:/mlflow]      # → EBS
    networks: [nvda-net]
    restart: unless-stopped

  nginx:
    image: nginx:1.27-alpine
    ports: ["80:80", "443:443"]         # the ONLY published ports
    volumes:
      - ./nginx/nvda.conf:/etc/nginx/conf.d/default.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    networks: [nvda-net]
    restart: unless-stopped

networks: {nvda-net: {driver: bridge}}
volumes: {mlflow-data: {}}
```

```bash
docker compose -f deploy/docker-compose.yml up -d --build
curl localhost/health
```

---

## Results

**Test window:** 2022-03-18 → 2026-04-10 (1,019 trading days), strict one-step-ahead.

### Splits

| Split | Rows | Start | End | Close min | Close max |
|---|---:|---|---|---:|---:|
| train | 4,750 | 1999-04-19 | 2018-03-02 | $0.03 | $6.23 |
| val | 1,018 | 2018-03-05 | 2022-03-17 | $3.18 | $33.38 |
| test | 1,019 | 2022-03-18 | 2026-04-10 | $11.23 | $207.04 |

> The **33× level gap** between the training and test maxima is the single most consequential fact in this project. It is what makes level-based modelling fail and forces the return-based target.

### Model Comparison

| Model | RMSE | MAE | MAPE | Directional Acc. | Skill vs naive | DM p-value | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **drift** | **2.998** | **1.914** | **2.382** | 53.83% | +0.077% | 0.622 | < 1s |
| lstm | 2.998 | 1.914 | 2.384 | 53.54% | +0.055% | 0.737 | ~20s |
| xgboost | 2.999 | 1.917 | 2.382 | 52.16% | +0.027% | 0.938 | < 5s |
| naive | 3.000 | 1.919 | 2.386 | undefined | — | — | instant |
| arima | 4.083 | 2.704 | 3.423 | 49.02% | −36.1% | < 0.001 | ~45s |
| ma5 | 4.230 | 2.856 | 3.619 | 49.41% | −41.0% | < 0.001 | instant |
| prophet | 26.20 | 19.78 | 27.22 | 50.10% | −773% | < 0.001 | ~115s |

**Up-day base rate in the test window: 53.78%** — the number any directional accuracy must beat.

### Walk-Forward Validation (16 rolling quarters)

| Model | RMSE mean | RMSE std | Folds beaten naive | Binomial p |
|---|---:|---:|---:|---:|
| drift | 2.513 | 1.675 | 10 / 16 | 0.454 |
| xgboost | 2.514 | 1.672 | 10 / 16 | 0.454 |
| naive | 2.515 | 1.673 | — | — |
| arima | 3.452 | 2.200 | 0 / 16 | < 0.001 |
| ma5 | 3.575 | 2.280 | 0 / 16 | < 0.001 |

Ten wins of sixteen; a coin flip gives eight. XGBoost's optimal tree count across folds: 92, 208, 58, 198, 236, 181, 61, 15, 44, 44, 41, 43, 43, 42, 39, 14. A model whose optimal complexity swings seventeen-fold between adjacent quarters is fitting noise.

### Drift Detection

**4 of 46 features drifted significantly — all volatility or range measures, all shifted downward.**

| Feature | PSI | Mean shift (σ) |
|---|---:|---:|
| volatility_21 | 1.483 | −0.139 |
| ATR14_norm | 0.808 | −0.222 |
| log_vol_change | 0.271 | −0.004 |
| vol_over_MA20 | 0.262 | −0.041 |

Training spans the dot-com collapse and 2008; the test window does not. Momentum and moving-average ratios are all stable — trend structure is unchanged, only amplitude moved.

The EDA notebook establishes the same thing from the raw series: Ljung-Box on returns gives **p = 0.053**, on squared returns **p = 3 × 10⁻¹¹⁹**. Direction is unpredictable; magnitude is not. That is the argument for modelling volatility next.

### Reading the table

- **drift, lstm, xgboost and naive are one cluster.** RMSEs differ in the third decimal; DM p-values above 0.6 say the differences are noise.
- **Naive has no directional accuracy.** It predicts zero change every day, so it never commits to a direction. Reporting 0% would mislead.
- **drift wins because NVDA went up.** Its 53.83% is indistinguishable from the 53.78% base rate — it predicts up every day.
- **arima and ma5 are significantly worse**, and lose in every single walk-forward fold.
- **prophet fails informatively.** A 27% MAPE is what structural model mismatch looks like.

Full analysis in [`reports/EVALUATION.md`](reports/EVALUATION.md).

---

## Live Resources

| Service | Location |
|---|---|
| S3 data lake — raw | `s3://<bucket>/raw/nvda/ingest_date=YYYY-MM-DD/` |
| S3 data lake — processed | `s3://<bucket>/processed/nvda/year=YYYY/month=M/` |
| S3 — MLflow artifacts | `s3://<bucket>/mlflow-artifacts/` |
| FastAPI Swagger UI | `http://localhost:8000/docs` (or `http://<host>/docs`) |
| FastAPI Health | `http://localhost:8000/health` |
| MLflow UI | `http://localhost:5002` (or `http://<host>/mlflow/`) |

> No public EC2 endpoint is listed because none is deployed. The S3 resources above are live in ap-south-1.

---

## Limitations

- **Horizon is one day.** Multi-step forecasting needs recursive prediction with error accumulation, or direct multi-horizon models.
- **No transaction costs or slippage.** Directional accuracy is not a trading strategy — 53.8% at retail spreads is not profitable.
- **Single asset, single regime.** The test window spans one exceptional bull run.
- **EC2, EMR and CloudWatch are written but never executed.** A t3.large costs roughly $60/month, and a screenshot of an instance terminated the same afternoon would not make them more true.
- **A 1999–2018 training window teaches the model about a company that no longer exists in the same form.**

### What would actually be worth trying

1. **Change the target.** Volatility clusters and is forecastable; returns are not. GARCH would likely be the first model here to show real skill.
2. **Change the horizon.** Weekly or monthly aggregation reduces microstructure noise.
3. **Change the features.** Options-implied volatility, earnings surprises or supply-chain data carry information not already in the price.
4. **Change the question.** "Will tomorrow's move exceed ±2%" is more tractable than predicting the price.

What is *not* worth trying: more technical indicators, deeper networks, more hyperparameter search.

---

## Disclaimer

Research and educational use only. **Nothing here is investment advice.** A next-day price forecast — particularly one that does not beat a random walk — must not be used to make trading decisions.

---

## License

MIT License — feel free to fork and build on this.

---

## Author

Built and deployed end-to-end as a demonstration of a production ML system — from data ingestion and causal feature engineering, through multi-model training with honest baselines and statistical significance testing, to experiment tracking, containerized serving, and an AWS S3 data lake with least-privilege IAM.

**Stack:** Python · PyTorch · XGBoost · Prophet · statsmodels · FastAPI · MLflow · PySpark · Docker · Nginx · Gunicorn · AWS S3/IAM · GitHub Actions
