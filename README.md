# 📈 NVIDIA Stock Price Forecasting

> End-to-end time-series ML pipeline for forecasting NVIDIA (NVDA) stock prices using ARIMA, Prophet, XGBoost, and LSTM — benchmarked against random-walk baselines, served via FastAPI, tracked with MLflow, and deployed on AWS EC2 with an S3 data lake.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3-orange?logo=amazonaws&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen)

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
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [AWS S3 Data Lake](#aws-s3-data-lake)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Configuration](#docker-configuration)
- [Results](#results)
- [Live Endpoints](#live-endpoints)

---

## Overview

This project builds a production-grade stock price forecasting system for NVIDIA (ticker: `NVDA`). It trains four time-series and ML models plus three baselines, exposes predictions through a REST API, tracks all experiments using MLflow with a Model Registry, and is deployed on AWS EC2 with an S3 data lake.

The system supports:
- Historical stock price ingestion, validation and causal feature engineering
- Multi-model training pipeline (ARIMA, Prophet, XGBoost, LSTM) plus naive / MA / drift baselines
- Walk-forward validation and Diebold-Mariano significance testing
- REST API for real-time forecasting with an empirical prediction interval
- Data drift detection and prediction-distribution monitoring
- MLflow UI for experiment tracking, model comparison and stage promotion
- AWS S3 data lake with immutable raw zone and year/month Parquet partitions
- Deployment on AWS EC2 (Mumbai region), with a containerized stack for production

**Dataset:** 6,847 daily bars, 1999-01-22 → 2026-04-13. Zero missing values, zero duplicate dates.

---

## Key Finding

> **No model in this project beats a random walk to any statistically significant degree.**

On a 1,019-day held-out window the best model improves RMSE over "tomorrow's price equals today's price" by **0.08%**, with a Diebold-Mariano p-value of **0.62**. Walk-forward validation across 16 rolling quarters puts its win rate at 10/16 — a coin flip gives 8, binomial p = 0.454.

The apparent 53.8% directional accuracy is fully explained by the base rate of up-days in the test window (**53.78%**): the winning model predicts *up* every single day.

This is the correct result for daily equity closes, not a pipeline failure. Projects reporting sub-1% MAPE without a baseline row are usually measuring a lagged copy of the input — which is exactly what the `naive` row quantifies.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER                             │
└────────────────────┬───────────────────────┬────────────────────────┘
                     │                       │
                     ▼                       ▼
           Port 8000 (FastAPI)       Port 5002 (MLflow UI)
                     │                       │
┌────────────────────▼───────────────────────▼────────────────────────┐
│                       AWS EC2 (t3.micro)                            │
│                        Ubuntu 26.04 LTS                             │
│                      ap-south-1 (Mumbai)                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Application Layer                          │   │
│  │                                                              │   │
│  │   ┌────────────────────┐        ┌────────────────────────┐   │   │
│  │   │      nvda-api      │        │      nvda-mlflow       │   │   │
│  │   │  FastAPI + uvicorn │───────▶│   MLflow Server        │   │   │
│  │   │       :8000        │        │   + Model Registry     │   │   │
│  │   │                    │        │        :5002           │   │   │
│  │   │  /forecast         │        │                        │   │   │
│  │   │  /health           │        │  sqlite:///mlflow.db   │   │   │
│  │   │  /predictions      │        │                        │   │   │
│  │   │  /drift            │        └────────────────────────┘   │   │
│  │   │  /metrics          │                                     │   │
│  │   └─────────┬──────────┘                                     │   │
│  │             │                                                │   │
│  │      models/best_model.pkl                                   │   │
│  └─────────────┼────────────────────────────────────────────────┘   │
└────────────────┼────────────────────────────────────────────────────┘
                 │
                 ▼
      ┌──────────────────────────────────────────┐
      │                 AWS S3                   │
      │   raw/  ·  processed/  ·  mlflow-        │
      │   artifacts/   versioned · lifecycle     │
      │   least-privilege IAM                    │
      └──────────────────────────────────────────┘

Security Group Inbound Rules:
  22    TCP   My IP        SSH
  8000  TCP   0.0.0.0/0    FastAPI
  5002  TCP   0.0.0.0/0    MLflow UI
```

### Data Flow

```
NVIDIA OHLCV (Yahoo Finance / cached snapshot)
      │  6,847 rows · 1999-01-22 → 2026-04-13
      ▼
① Data Ingestion (yfinance)
      │  retries w/ backoff · schema validation · offline CSV fallback
      ▼
② Land raw in S3 ────────────▶ raw/nvda/ingest_date=YYYY-MM-DD/
      │                         immutable · CSV / JSON / Parquet
      ▼
③ Preprocessing
      │  dedupe · sort · forward-fill · zero-volume repair
      │  NO calendar reindex (weekends are not missing data)
      ▼
④ Feature Engineering ───────▶ 46 causal, scale-free features
      ├── MA ratios      close_over_MA7 / MA20 / MA50
      ├── Oscillators    RSI14, MACD_norm, BB_pctB
      ├── Volatility     volatility_7 / _21, ATR14_norm
      ├── Lags           log_ret_lag_{1,2,3,5,7,14,30}
      ├── Volume         vol_over_MA20, dollar_vol_z
      └── Calendar       dow, month, is_month_end
      ▼
⑤ Publish processed ─────────▶ processed/nvda/year=YYYY/month=M/
      │                         325 Parquet partitions
      ▼
⑥ Chronological Split (70 / 15 / 15) — no shuffling, no leakage
      │  train 4,750 │ val 1,018 │ test 1,019
      ▼
⑦ Model Training
      │  ARIMA · Prophet · XGBoost · LSTM
      │  + BASELINES: naive · ma5 · drift
      ▼
⑧ Evaluation — RMSE · MAE · MAPE · Directional Accuracy
      │           · Diebold-Mariano · walk-forward (16 folds)
      ▼
⑨ MLflow Tracking → Model Registry → approval gate → FastAPI
```

---

## Features

- **Multi-model forecasting** — ARIMA, Prophet, XGBoost, LSTM in one unified harness
- **Baselines that keep models honest** — naive (random walk), 5-day MA, drift, scored identically
- **Statistical significance** — Diebold-Mariano with Newey-West variance and Harvey correction
- **Walk-forward validation** — 16 expanding-window folds, each model refitted from scratch
- **Return-based targets** — models learn log returns, not price levels
- **Leakage controls with tests** — 7 explicit assertions
- **REST API** — FastAPI with Swagger UI and an empirical 80% prediction interval
- **Drift detection** — PSI and KS tests on every feature, served live at `/drift`
- **Prediction monitoring** — distribution of recent forecasts with an implausible-move alarm
- **Experiment tracking** — MLflow with Model Registry and a three-gate approval step
- **AWS S3 data lake** — immutable raw zone, year/month partitions, lifecycle rules, least-privilege IAM
- **Deployed on EC2** — live public endpoints in ap-south-1

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI + Uvicorn (Gunicorn in the container) |
| ML Models | statsmodels (ARIMA), Prophet, XGBoost, PyTorch (LSTM) |
| Baselines | Naive / Moving Average / Drift |
| Statistics | scipy (Diebold-Mariano, PSI, KS), statsmodels |
| Experiment Tracking | MLflow 3.x (SQLite backend + Model Registry) |
| Data Source | yfinance (Yahoo Finance) + cached CSV snapshot |
| Feature Engineering | pandas, numpy (hand-rolled indicators) |
| Big Data | PySpark 4.x (local + AWS EMR) |
| Cloud Storage | AWS S3 (boto3, s3fs) |
| Testing | pytest (110 tests), moto, ruff |
| Containerization | Docker (multi-stage) + Docker Compose |
| Web Server | Nginx (reverse proxy, TLS, rate limiting) |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2 (t3.micro), S3, IAM |
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
│       ├── cloudwatch_alarms.json   # 7 alarms + SNS topic
│       └── s3_layout.md             # Data-lake layout + lifecycle policy
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── .github/workflows/ci-cd.yml
├── tests/                           # 110 tests
├── data/raw/nvda_raw.csv
├── models/                          # arima/prophet/xgb/lstm + best_model.pkl
├── reports/
│   ├── EVALUATION.md
│   ├── metrics.csv
│   ├── walkforward_folds.csv
│   └── figures/                     # 8 PNGs
│
├── Makefile
├── requirements.txt
└── requirements-serve.txt           # Inference only (no torch/prophet)
```

---

## Models

### 0. Baselines — naive / ma5 / drift

The row most stock-prediction projects omit. `naive` is `ŷ(t+1) = y(t)` — the random walk, and under weak-form market efficiency the theoretically optimal point forecast.

- **Speed:** instant
- **Why it matters:** a model reporting 2% MAPE that does not beat this has learned nothing

### 1. ARIMA

Classical statistical model fitted on `log(Close)` with `d=1`, making it an ARMA on log returns — the stationary representation.

- **Params:** `order=(2,1,2)`, linear trend; `grid_search_order()` selects by AIC
- **Backtest:** `append(refit=False)` — true one-step-ahead with fixed coefficients
- **Speed:** ~45s

### 2. Facebook Prophet

Additive trend + seasonality decomposition. Included per the brief, and the result is informative: it shows what happens when a smooth trend model meets a near-martingale.

- **Params:** `changepoint_prior_scale=0.10`, weekly + yearly seasonality
- **Backtest:** retrained every 63 trading days (~quarterly), rolled forward
- **Speed:** ~115s (16 refits)

### 3. XGBoost

Gradient-boosted trees on 46 engineered features. **The target is `log(Close_t+1 / Close_t)`, not the price.**

> ⚠️ The chronological split puts the training maximum at **$6.23** while the test window reaches **$207.04**. Trees predict the mean of a leaf, so output is bounded by the training target range. A price-level XGBoost would emit a flat line near $6 for the entire test period — RMSE around $70 — while appearing to have trained successfully.

- **Params:** `n_estimators=600`, `max_depth=4`, `lr=0.02`, early stopping on validation
- **Speed:** < 5s

### 4. LSTM (PyTorch)

Sequence model over a 30-day window of the same stationary features. Same target choice — a min-max scaler fitted on training prices maps every test price above 1.0.

- **Architecture:** 2-layer LSTM (48 hidden) → Dropout → Dense(32) → Output
- **Params:** `lookback=30`, `dropout=0.2`, `lr=1e-3`, Huber loss, early stopping
- **Speed:** ~20s on CPU

---

## API Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "drift",
  "trained_at": "2026-09-13T04:23:26.977303+00:00",
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

> The 80% interval is computed from realised 60-day volatility, **not** from the model. None of these point forecasters produces a calibrated predictive distribution.

### `GET /model`

Deployed model metadata and its held-out metrics, including the naive RMSE it should be judged against.

### `GET /predictions`

Distribution of recent forecasts. `implausible_rate_pct` is the alarm signal — latency says the service is up, this says whether it has started emitting nonsense.

### `GET /drift`

Live PSI / KS data drift against the training distribution.

### `GET /metrics`

Prometheus text format: request counts, errors, latency, prediction gauges.

---

## Local Setup

### Prerequisites

- Python 3.12+, Git, (optional) Docker Desktop

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Priyankaakrish/Stock-Market-Prediction-Using-Time-Series.git
cd Stock-Market-Prediction-Using-Time-Series

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
python -m pytest tests -q       # 110 tests

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

## AWS EC2 Deployment

### Infrastructure

| Setting | Value |
|---|---|
| Instance Type | t3.micro (free tier eligible) |
| OS | Ubuntu 26.04 LTS |
| Region | ap-south-1 (Mumbai) |
| Storage | 8 GiB gp3 |
| Public IP | 13.233.91.41 |

### Step 1 — Create Security Group

EC2 → Security Groups → Create security group. Name `nvda-sg`, with these inbound rules:

```
Type          Port    Source
SSH           22      My IP
Custom TCP    8000    0.0.0.0/0     (FastAPI)
Custom TCP    5002    0.0.0.0/0     (MLflow UI)
```

> SSH is restricted to your own address. Port 22 open to the world attracts credential-stuffing bots within minutes.

### Step 2 — Launch EC2 Instance

EC2 → Launch instance:

```
Name              nvda-forecasting
AMI               Ubuntu Server 26.04 LTS
Instance type     t3.micro
Key pair          nvda-key (.pem, RSA) — downloads once, save it
Security group    Select existing → nvda-sg
Storage           8 GiB gp3
```

### Step 3 — Connect via SSH

```bash
# Windows (PowerShell) — fix key permissions first
icacls nvda-key.pem /inheritance:r
icacls nvda-key.pem /grant:r "$($env:USERNAME):(R)"
ssh -i nvda-key.pem ubuntu@13.233.91.41

# Linux/Mac
chmod 400 nvda-key.pem
ssh -i nvda-key.pem ubuntu@13.233.91.41
```

> If SSH times out, your ISP has rotated your IP. Edit the security group's SSH rule and re-select **My IP**.

### Step 4 — Install System Packages

```bash
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip git
```

### Step 5 — Clone and Install

```bash
git clone https://github.com/Priyankaakrish/Stock-Market-Prediction-Using-Time-Series.git repo
cd repo
python3 -m venv venv && source venv/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir numpy pandas scipy statsmodels fastapi "uvicorn[standard]" pydantic
```

> Serve-only dependencies. The full `requirements.txt` pulls PyTorch, which drags in ~2 GB of CUDA packages onto a box with no GPU and 8 GB of disk. `--no-cache-dir` stops pip keeping a second copy of every wheel.

### Step 6 — Build the Dataset and Model

`data/processed/` and `models/*.pkl` are gitignored, so the instance builds its own:

```bash
python -m src.ingest && python -m src.preprocess && python -m src.features && python -m src.split

python -c "
import json, datetime
from src.split import load_splits
from src.models.baselines import DriftForecaster
from src.config import PATHS
s = load_splits()
DriftForecaster().fit(s.train).save(PATHS.best_model)
(PATHS.models / 'best_model_meta.json').write_text(json.dumps({
 'model':'drift','trained_at':datetime.datetime.now(datetime.UTC).isoformat(),
 'horizon_days':1,'metrics':{'rmse':2.9977,'mae':1.9136,'mape':2.3818,
 'directional_accuracy':53.83,'r2':0.9977},'naive_rmse':3.0}, indent=2))
print('champion saved')
"
```

### Step 7 — Start the API

```bash
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
sleep 8 && curl localhost:8000/health
```

> `--host 0.0.0.0` is essential. The default binds to localhost only, and nothing outside the instance can reach it. `nohup ... &` detaches the process so it survives the SSH session ending.

### Step 8 — Start MLflow (optional)

```bash
pip install --no-cache-dir mlflow
nohup mlflow server --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 --port 5002 > mlflow.log 2>&1 &
```

To populate it with local experiment history, from your machine:

```bash
scp -i nvda-key.pem mlflow.db ubuntu@13.233.91.41:~/repo/
scp -i nvda-key.pem -r mlartifacts ubuntu@13.233.91.41:~/repo/
```

### Step 9 — Verify Deployment

```bash
# Test FastAPI
curl http://localhost:8000/health

# Test from browser
# FastAPI Docs:   http://13.233.91.41:8000/docs
# FastAPI Health: http://13.233.91.41:8000/health
# MLflow UI:      http://13.233.91.41:5002
```

### Expand EBS Volume (if disk full)

8 GiB fills quickly. If a `pip install` fails with `No space left on device`:

```bash
# After expanding the volume in the AWS Console:
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h  # Verify new size
```

Quick wins without expanding:

```bash
sudo swapoff /swapfile && sudo rm /swapfile   # a 4 GB swapfile on 8 GB of disk
rm -rf ~/.cache/*
sudo apt-get clean
```

### After EC2 Reboot

The public IP changes on stop/start unless an Elastic IP is assigned, and the detached processes do not survive a reboot:

```bash
cd ~/repo && source venv/bin/activate
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
nohup mlflow server --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 --port 5002 > mlflow.log 2>&1 &
```

> **Tip:** Assign an AWS Elastic IP to keep a permanent public address, and add a systemd unit if you want the services to restart automatically.

---

## AWS S3 Data Lake

| Component | State |
|---|---|
| Raw zone (CSV / JSON / Parquet) | `raw/nvda/ingest_date=YYYY-MM-DD/`, immutable |
| Processed zone | 325 `year=/month=` Parquet partitions, 16 MB |
| MLflow artifact store | `mlflow-artifacts/models/…/python_model.pkl` |
| Versioning + lifecycle | 4 rules: Glacier IR, Standard-IA, multipart abort |
| IAM | Least privilege, scoped to three prefixes |

```bash
export S3_BUCKET=<your-bucket>
export AWS_REGION=ap-south-1
python -m src.storage keys        # prints the layout, writes nothing
python -m src.storage upload      # raw + 325 partitions
python -m src.pipeline --mode s3 --fast --promote
```

> `aws s3 ls s3://bucket/ --recursive` fails with AccessDenied while `aws s3 ls s3://bucket/raw/` succeeds, because the policy grants `ListBucket` only under a prefix condition. Least privilege you can watch refuse something is least privilege that actually exists.

---

## MLflow Experiment Tracking

MLflow UI is accessible at: `http://13.233.91.41:5002`

Each training run logs:

**Parameters tracked:**
- Model type (arima/prophet/xgboost/lstm/naive/ma5/drift)
- Hyperparameters per model, plus target representation (`log_return` vs `log_close`)
- Feature count, split boundaries, train/test date ranges

**Metrics tracked:**
- RMSE, MAE, MAPE (validation and test)
- Directional accuracy, R², fit seconds
- `test_up_day_base_rate`, `naive_test_rmse`, `best_skill_vs_naive_pct`

**Artifacts stored:**
- Trained model file per model
- 8 figures (splits, predictions, comparison, importance, residuals, LSTM curve, walk-forward)
- `metrics.csv`, `test_predictions.csv`, `feature_importance.csv`

### Experiment Structure

```
nvda-price-forecasting/
└── pipeline-20260913-042326/
    ├── naive     → rmse=3.000  mape=2.39%  DA=n/a
    ├── ma5       → rmse=4.230  mape=3.62%  DA=49.41%
    ├── drift     → rmse=2.998  mape=2.38%  DA=53.83%   ★ champion
    ├── arima     → rmse=4.083  mape=3.42%  DA=49.02%
    ├── prophet   → rmse=26.20  mape=27.22% DA=50.10%
    ├── xgboost   → rmse=2.999  mape=2.38%  DA=52.16%
    └── lstm      → rmse=2.998  mape=2.38%  DA=53.54%
```

The champion is registered as `nvda-forecaster` with Staging and Production aliases:

```bash
python -m src.promote list       # every version and its alias
python -m src.promote check      # evaluate the newest against the gate
python -m src.promote promote    # set the Production alias
```

Three gates: not materially worse than the incumbent, MAPE under 5%, and **not worse than a random walk**.

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

> The serving image excludes torch and prophet — shipping a 2 GB CUDA stack to serve a model that is a few hundred bytes of coefficients is waste.

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
    volumes: [mlflow-data:/mlflow]
    networks: [nvda-net]
    restart: unless-stopped

  nginx:
    image: nginx:1.27-alpine
    ports: ["80:80", "443:443"]         # the ONLY published ports
    volumes:
      - ./nginx/nvda.conf:/etc/nginx/conf.d/default.conf:ro
    networks: [nvda-net]
    restart: unless-stopped

networks: {nvda-net: {driver: bridge}}
volumes: {mlflow-data: {}}
```

```bash
docker compose -f deploy/docker-compose.yml up -d --build
curl localhost/health
```

> The containerized stack is validated locally. The t3.micro deployment above runs uvicorn directly, because building the image needs more RAM and disk than a free-tier instance has.

---

## Results

**Test window:** 2022-03-18 → 2026-04-10 (1,019 trading days), strict one-step-ahead.

### Splits

| Split | Rows | Start | End | Close min | Close max |
|---|---:|---|---|---:|---:|
| train | 4,750 | 1999-04-19 | 2018-03-02 | $0.03 | $6.23 |
| val | 1,018 | 2018-03-05 | 2022-03-17 | $3.18 | $33.38 |
| test | 1,019 | 2022-03-18 | 2026-04-10 | $11.23 | $207.04 |

> The **33× level gap** between the training and test maxima is what makes level-based modelling fail and forces the return-based target.

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

XGBoost's optimal tree count across folds: 92, 208, 58, 198, 236, 181, 61, 15, 44, 44, 41, 43, 43, 42, 39, 14. A model whose optimal complexity swings seventeen-fold between adjacent quarters is fitting noise.

### Drift Detection

**4 of 46 features drifted significantly — all volatility or range measures, all shifted downward.**

| Feature | PSI | Mean shift (σ) |
|---|---:|---:|
| volatility_21 | 1.483 | −0.139 |
| ATR14_norm | 0.808 | −0.222 |
| log_vol_change | 0.271 | −0.004 |
| vol_over_MA20 | 0.262 | −0.041 |

Training spans the dot-com collapse and 2008; the test window does not. Momentum and moving-average ratios are all stable — trend structure is unchanged, only amplitude moved.

The EDA notebook establishes the same from the raw series: Ljung-Box on returns gives **p = 0.053**, on squared returns **p = 3 × 10⁻¹¹⁹**. Direction is unpredictable; magnitude is not.

> drift wins because NVDA went up. Its 53.83% is indistinguishable from the 53.78% base rate — it predicts up every day. Full analysis in [`reports/EVALUATION.md`](reports/EVALUATION.md).

---

## Live Endpoints

| Service | URL |
|---|---|
| FastAPI Swagger UI | http://13.233.91.41:8000/docs |
| FastAPI Health | http://13.233.91.41:8000/health |
| FastAPI Forecast (POST) | http://13.233.91.41:8000/forecast |
| MLflow UI | http://13.233.91.41:5002 |

> Hosted on a t3.micro in ap-south-1. If the links do not resolve, the instance has been stopped to preserve free-tier credits — every command needed to bring it back is in [AWS EC2 Deployment](#aws-ec2-deployment).

---

## Limitations

- **Horizon is one day.** Multi-step forecasting needs recursive prediction with error accumulation.
- **No transaction costs or slippage.** Directional accuracy is not a trading strategy.
- **Single asset, single regime.** The test window spans one exceptional bull run.
- **EMR and CloudWatch are written but never executed.** Both need infrastructure beyond the free tier.
- **A 1999–2018 training window teaches the model about a company that no longer exists in the same form.**

### What would actually be worth trying

1. **Change the target.** Volatility clusters and is forecastable; returns are not. GARCH would likely be the first model here to show real skill.
2. **Change the horizon.** Weekly or monthly aggregation reduces microstructure noise.
3. **Change the features.** Options-implied volatility, earnings surprises, supply-chain data.
4. **Change the question.** "Will tomorrow's move exceed ±2%" is more tractable than predicting the price.

---

## Disclaimer

Research and educational use only. **Nothing here is investment advice.** A next-day price forecast — particularly one that does not beat a random walk — must not be used to make trading decisions.

---

## License

MIT License — feel free to fork and build on this.

---

## Author

Built and deployed end-to-end as a demonstration of a production ML system — from data ingestion and causal feature engineering, through multi-model training with honest baselines and statistical significance testing, to experiment tracking, containerized serving, an AWS S3 data lake with least-privilege IAM, and a live EC2 deployment.

**Stack:** Python · PyTorch · XGBoost · Prophet · statsmodels · FastAPI · MLflow · PySpark · Docker · Nginx · AWS EC2/S3/IAM · GitHub Actions
