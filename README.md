# 📈 Goldman Sachs Stock Price Prediction

> End-to-end ML pipeline for forecasting Goldman Sachs (GS) stock prices and volatility using a naive baseline, ARIMA, Prophet, XGBoost, LSTM, and GARCH — served via FastAPI, tracked with MLflow, and deployed on AWS EC2 with Docker Compose.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-3.x-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)
![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Data Integrity Finding](#data-integrity-finding)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Models](#models)
- [Evaluation Protocol](#evaluation-protocol)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Configuration](#docker-configuration)
- [Testing](#testing)
- [Results](#results)
- [Live Endpoints](#live-endpoints)

---

## Overview

This project builds a production-grade forecasting system for Goldman Sachs (ticker: `GS`). It trains six model families across two targets, exposes predictions through a REST API, tracks all experiments using MLflow, and is fully containerized and deployable on AWS EC2.

The system supports:
- Historical price ingestion with **schema validation that fails loudly**
- Multi-model training pipeline (naive baseline, ARIMA, Prophet, XGBoost, LSTM)
- Volatility pipeline (GARCH, GJR-GARCH, Student-t variants)
- **Baseline-relative scoring** so a good-looking MAPE cannot be mistaken for a good model
- REST API for forecasting with configurable horizons
- MLflow UI for experiment tracking and model comparison
- Docker-based deployment on AWS EC2

**What this project claims.** It builds a correct, reproducible pipeline and reports what the models actually achieve. On price, **no model beats a random walk** — that is the finding, and it appears in the Results table rather than being hidden. Every model is therefore scored on **SKILL**:

```
SKILL = 1 - RMSE_model / RMSE_naive
```

`SKILL > 0` beats the random walk. `SKILL <= 0` does not.

---

## Data Integrity Finding

The supplied `goldmansachs.csv` ships with **shifted column headers**. In 6,596 of 6,709 rows the `low` column exceeds the `high` column, which is impossible for a real price bar.

| Header in file | Actual field |
|---|---|
| `open` | Adjusted Close |
| `high` | Close |
| `low` | **High** |
| `close` | **Low** |
| `adj_close` | Open |

Two independent confirmations:

1. Under this mapping **all 6,709 rows** satisfy `High >= max(Open, Close)` and `Low <= min(Open, Close)`. No other permutation comes close.
2. `open / high` rises smoothly from 0.692 (1999) to exactly 1.000 (2026) — the signature of a dividend-adjustment factor converging, which only makes sense if those two columns are Adj Close and Close.

`src/data/loader.py` applies the repair, then **re-validates and raises** if any row still fails. Dataset after repair: **6,709 trading days, 1999-05-04 to 2026-01-02**, zero nulls, zero duplicate dates.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER                             │
└────────────────────┬───────────────────────┬────────────────────────┘
                     │                       │
                     ▼                       ▼
           Port 8000 (FastAPI)       Port 5000 (MLflow UI)
                     │                       │
┌────────────────────▼───────────────────────▼────────────────────────┐
│                        AWS EC2 (t3.small)                           │
│                       Ubuntu 24.04 LTS                              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Docker Network (gs-net)                   │   │
│  │                                                              │   │
│  │   ┌─────────────────────┐    ┌──────────────────────────┐    │   │
│  │   │     gs-api          │    │      gs-mlflow           │    │   │
│  │   │   (FastAPI +        │───►│   (MLflow Server)        │    │   │
│  │   │    Gunicorn)        │    │                          │    │   │
│  │   │                     │    │  --host 0.0.0.0          │    │   │
│  │   │  • naive / drift    │    │  sqlite backend store    │    │   │
│  │   │  • ARIMA model      │    │  + Model Registry        │    │   │
│  │   │  • Prophet model    │    │                          │    │   │
│  │   │  • XGBoost model    │    │  Port 5000 published     │    │   │
│  │   │  • LSTM model       │    │  directly — no proxy     │    │   │
│  │   │  • GARCH models     │    │                          │    │   │
│  │   └─────────────────────┘    └──────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │              EBS Volume (20 GB)                              │  │
│   │   ~/gs_stock_prediction/mlruns  (SQLite + artifact store)    │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

AWS Security Group Inbound Rules:
  ┌──────────┬──────────┬─────────────┐
  │ Port     │ Protocol │ Source      │
  ├──────────┼──────────┼─────────────┤
  │ 22       │ TCP      │ My IP       │
  │ 8000     │ TCP      │ 0.0.0.0/0   │
  │ 5000     │ TCP      │ My IP       │
  └──────────┴──────────┴─────────────┘
```

> **No Nginx and no TCP proxy.** Those are only needed when MLflow binds to `127.0.0.1` inside its container. Starting it with `--host 0.0.0.0` and publishing the port removes both components. Port 5000 is restricted to your own IP because **MLflow has no authentication**.

### Data Flow

```
data/raw/goldmansachs.csv
      │
      ▼
Data Loading (src/data/loader.py)
  ├── Header repair (shifted columns)
  ├── OHLC invariant validation  ──► RAISES on failure
  └── Adjusted Close as target
      │
      ▼
Feature Engineering (src/features/engineer.py)
  ├── Lagged returns (t-1 … t-21)
  ├── RSI, MACD, Bollinger Bands, ATR
  ├── Rolling mean / std / momentum
  └── Volume ratio — ALL .shift(1)-ed
      │
      ▼
Chronological Split (never shuffled)
  train 5,063 │ val 893 │ test 753
      │
      ▼
┌───────────────────────────────────────────────┐
│              Training Pipeline                │
│  ┌─────────────────────────────────────────┐  │
│  │  NAIVE BASELINE (random walk)           │  │
│  │  every model below is scored against it │  │
│  └─────────────────────────────────────────┘  │
│  ┌─────────┐  ┌──────────────────┐            │
│  │  ARIMA  │  │     Prophet      │            │
│  └────┬────┘  └────────┬─────────┘            │
│  ┌────▼────┐  ┌────────▼─────────┐            │
│  │ XGBoost │  │      LSTM        │            │
│  └────┬────┘  └────────┬─────────┘            │
│  ┌────▼───────────────────────────┐           │
│  │ GARCH / GJR-GARCH (volatility) │           │
│  └────┬───────────────────────────┘           │
└───────┼───────────────────────────────────────┘
        ▼
  Evaluation (src/evaluation/metrics.py)
  RMSE · MAE · MAPE · R² · Directional Acc · IC
  Sharpe · Max Drawdown · SKILL · QLIKE · VaR
        │
        ▼
  LEAKAGE GATE — flags any model scoring
  SKILL > 0.30 or directional accuracy > 65%
        │
        ▼
  MLflow Tracking (params, metrics, artifacts)
        │
        ▼
  FastAPI Service
        │
   ┌────┴─────┐
   ▼          ▼
/forecast   /health
/models
```

---

## Features

- **Six model families** — naive/drift baseline, ARIMA, Prophet, XGBoost, LSTM, GARCH volatility models
- **Baseline-relative scoring** — SKILL against a random walk on every model
- **Automatic leakage detection** — flags any result too good to be real, backed by tests asserting target alignment
- **Data validation that fails loudly** — OHLC invariants raise rather than warn
- **Walk-forward one-step evaluation** — parameters fitted on training data only, never refitted on the future
- **Volatility forecasting** — scored on QLIKE with Mincer-Zarnowitz regression and VaR calibration
- **REST API** — FastAPI with Swagger UI at `/docs`; every response carries its own backtest metrics
- **Experiment tracking** — MLflow logs parameters, metrics and artifacts per run
- **Model registry** — XGBoost booster registered with an inferred signature
- **Containerized** — Docker Compose orchestrates API and MLflow services
- **Tested** — 23 tests covering data integrity, leakage, model behaviour and API contracts

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI + Gunicorn + Uvicorn |
| **Price Models** | statsmodels (ARIMA), Prophet, XGBoost, PyTorch (LSTM) |
| **Volatility Models** | arch (GARCH, GJR-GARCH) |
| **Evaluation** | Custom metrics: SKILL, QLIKE, Mincer-Zarnowitz, VaR breach |
| **Experiment Tracking** | MLflow 3.x, SQLite backend |
| **Feature Engineering** | pandas, numpy (indicators implemented in-repo) |
| **Data Source** | Static CSV 1999–2026, header repair applied on load |
| **Testing** | pytest |
| **Containerization** | Docker + Docker Compose |
| **Cloud** | AWS EC2 t3.small, Ubuntu 24.04, EBS 20 GB |

> Indicators are implemented directly in `src/features/engineer.py` rather than via **ta-lib**, which needs a C library compiled per platform and frequently fails to install on Windows.

---

## Project Structure

```
gs_stock_prediction/
│
├── api/
│   ├── main.py                  # FastAPI app entry point
│   └── schemas.py               # Pydantic request/response models
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Header repair + OHLC validation
│   │   └── preprocessor.py      # Cleaning, temporal split, scaling
│   ├── features/
│   │   └── engineer.py          # Indicators, lags, target alignment
│   ├── models/
│   │   ├── baseline.py          # Naive + drift — the reference
│   │   ├── arima_model.py       # ARIMA wrapper + AIC grid search
│   │   ├── prophet_model.py     # Prophet wrapper
│   │   ├── xgboost_model.py     # XGBoost on log returns
│   │   ├── lstm_model.py        # PyTorch LSTM
│   │   └── garch_model.py       # GARCH family + volatility baselines
│   └── evaluation/
│       └── metrics.py           # Metrics + leakage detector
│
├── pipelines/
│   ├── train_pipeline.py        # Price models + MLflow logging
│   └── volatility_pipeline.py   # GARCH volatility forecasting
│
├── notebooks/
│   └── 01_EDA_Analysis.py       # Exploratory analysis
│
├── tests/
│   ├── test_data.py             # Integrity + leakage tests
│   ├── test_models.py           # Model behaviour
│   ├── test_volatility.py       # GARCH + QLIKE
│   └── test_api.py              # API contracts
│
├── deployment/
│   ├── Dockerfile               # API container image
│   └── docker-compose.yml       # Multi-container orchestration
│
├── mlruns/                      # MLflow store (auto-generated)
├── models_saved/                # Saved model files (auto-generated)
├── reports/figures/             # Plots (auto-generated)
├── data/
│   ├── raw/goldmansachs.csv
│   └── processed/               # Comparison tables (auto-generated)
│
├── config.py
├── requirements.txt
└── README.md
```

---

## Models

### 0. Naive / Drift Baseline
Random walk: tomorrow's price equals today's. **Not filler** — it is the reference every other model is scored against, and it is reported first.
- **Params:** optional drift term (mean per-step change)
- **Use case:** the bar every model must clear
- **Speed:** instant

### 1. ARIMA
Classical statistical time-series model on log prices. Captures autoregressive and moving-average components.
- **Params:** order `(p, d, q)` selected by AIC over 31 candidates, fitted on the most recent 1,500 observations
- **Selected:** `(2, 1, 3)`, AIC −7537.97
- **Forecasting:** walk-forward one-step via `.append(refit=False)` — parameters never see the future
- **Speed:** ~15s including grid search

### 2. Facebook Prophet
Additive forecasting model by Meta. Handles seasonality, holidays and trend changes automatically.
- **Params:** yearly/weekly seasonality, changepoint prior scale 0.05
- **Use case:** standard baseline — and a demonstration of structural mismatch on equity prices
- **Speed:** ~5s

### 3. XGBoost
Gradient boosted trees with engineered time-series features. **Target is the next log return, not the price.**
- **Params:** n_estimators 400, max_depth 4, learning_rate 0.03, early stopping on validation
- **Features:** lagged returns, RSI, MACD, Bollinger %, ATR, rolling volatility, momentum, volume ratio (35 total)
- **Why returns:** trees cannot extrapolate past their training range. GS ended training near $390 and testing near $914, so a price-level model is capped on every late prediction.
- **Speed:** < 1s

### 4. LSTM (PyTorch)
Deep sequence model over scaled log returns.
- **Architecture:** 2-layer LSTM (hidden 64, dropout 0.2) → Dense → Output
- **Params:** sequence_length 60, epochs 50 with early stopping
- **Discipline:** scaler fitted on **train only**; windows built first, then split by date; never shuffled
- **Speed:** slow on CPU

### 5. GARCH Family
Conditional volatility — the part of this series that is actually predictable.
- **Variants:** GARCH(1,1) normal, GARCH(1,1)-t, GJR-GARCH(1,1,1)-t
- **Baselines:** constant volatility, rolling 20/60-day, RiskMetrics EWMA (lambda 0.94)
- **Scored on:** QLIKE, Mincer-Zarnowitz R², 99% VaR breach rate
- **Speed:** < 1s

---

## Evaluation Protocol

**Split** — chronological, never shuffled:

| Split | Rows | Period | Purpose |
|---|---|---|---|
| Train | 5,063 | 1999-05-04 to 2019-06-17 | fitting |
| Validation | 893 | 2019-06-18 to 2022-12-30 | tuning, early stopping |
| Test | 753 | 2023-01-03 to 2026-01-02 | held out |

After tuning, ARIMA and Prophet are refitted on train + validation. Skipping this leaves the model's last known price years stale and poisons the first test forecast.

**Forecast mode** — walk-forward one-step-ahead. Predict day *t* using only information through day *t-1*.

**Target alignment** — every feature is `.shift(1)`-ed and the target at row *t* is day *t*'s own return. An early version used `logret.shift(-1)`, pairing row *t*'s features with the *t to t+1* return while the evaluator read the price at *t*. That produced **98.8% directional accuracy** before the misalignment was found. `check_for_leakage()` now fires on any SKILL > 0.30 or directional accuracy > 65%.

---

## API Endpoints

Base URL: `http://<EC2-PUBLIC-IP>:8000`

### `GET /health`
Returns API and data status.
```json
{
  "status": "healthy",
  "data_loaded": true,
  "n_observations": 6709,
  "date_range": ["1999-05-04", "2026-01-02"],
  "models_available": ["naive", "arima"]
}
```

### `POST /api/v1/forecast`
Get a stock price forecast.

**Request:**
```json
{
  "model": "arima",
  "ticker": "GS",
  "horizon": 30
}
```

**Response:**
```json
{
  "ticker": "GS",
  "model": "arima",
  "horizon": 30,
  "last_observed_date": "2026-01-02",
  "last_observed_price": 914.34,
  "forecast": [
    {"date": "2026-01-05", "predicted_price": 914.34,
     "lower_bound": 878.21, "upper_bound": 951.95}
  ],
  "backtest_metrics": {
    "RMSE": 8.9536,
    "MAPE": 1.2151,
    "Directional_Accuracy": 49.8008,
    "SKILL": -0.0103
  },
  "disclaimer": "Educational output. Not investment advice."
}
```

> The response carries `backtest_metrics` including **SKILL**, so no caller can consume a forecast without also seeing that the model does not beat a random walk.

### `GET /api/v1/models`
Lists all available models with their backtest RMSE and SKILL.

**Swagger UI:** `http://<EC2-PUBLIC-IP>:8000/docs`

---

## Local Setup

### Prerequisites
- Python 3.11+
- Git

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url> gs_stock_prediction
cd gs_stock_prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Explore the data
python notebooks/01_EDA_Analysis.py

# 5. Run training pipelines
python pipelines/train_pipeline.py --model all
python pipelines/volatility_pipeline.py

# 6. Run tests
pytest tests/ -q

# 7. Start MLflow UI locally
mlflow server --host 127.0.0.1 --port 5000 --workers 1 \
  --backend-store-uri sqlite:///mlruns/mlflow.db
# Open: http://localhost:5000

# 8. Start FastAPI locally
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000/docs
```

> **Windows note.** Pass `--workers 1` to `mlflow server`. Without it MLflow spawns a process pool that cannot share a listening socket on Windows, and a worker dies with `WinError 10022`.

> **PyTorch note.** Install from the CPU index first. The default PyPI wheel pulls ~2 GB of CUDA libraries you cannot use.

---

## AWS EC2 Deployment

### Infrastructure

| Setting | Value |
|---|---|
| AMI | Ubuntu Server 24.04 LTS |
| Instance type | **t3.small** (2 GB RAM) |
| Storage | 20 GB gp3 |
| Ports | 22, 8000, 5000 |

> **Use t3.small, not t3.micro.** 1 GB of RAM is not enough to `pip install` PyTorch and Prophet; the build gets OOM-killed. Set EBS to 20 GB — the 8 GB default fills during the Docker build.

### Step 1 — Launch EC2 Instance

Security group inbound: `22` from My IP, `8000` from `0.0.0.0/0`, `5000` from My IP.

### Step 2 — Connect via SSH

```powershell
# Windows (PowerShell)
icacls.exe ubuntu-key.pem /reset
icacls.exe ubuntu-key.pem /grant:r "$($env:USERNAME):(R)"
icacls.exe ubuntu-key.pem /inheritance:r
ssh -i ubuntu-key.pem ubuntu@<EC2-PUBLIC-IP>
```

```bash
# Linux/Mac
chmod 400 ubuntu-key.pem
ssh -i ubuntu-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3 — Upload Project to EC2

```powershell
# From your local machine
scp -i ubuntu-key.pem -r ./gs_stock_prediction ubuntu@<EC2-PUBLIC-IP>:~/
```

### Step 4 — Install Docker on EC2

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu && newgrp docker
docker --version && docker compose version
```

### Step 5 — Build and Start Containers

```bash
cd ~/gs_stock_prediction/deployment
docker compose up -d --build
docker compose ps
```

First build takes 10–15 minutes; Prophet compiles its Stan backend.

### Step 6 — Run Training Pipeline on EC2

```bash
docker compose exec gs-api python pipelines/train_pipeline.py --model all
docker compose exec gs-api python pipelines/volatility_pipeline.py
```

Results persist on the host through the mounted `../mlruns` and `../data` volumes.

### Step 7 — Verify Deployment

```bash
curl http://localhost:8000/health
```

From a browser:
- FastAPI Docs: `http://<EC2-PUBLIC-IP>:8000/docs`
- MLflow UI: `http://<EC2-PUBLIC-IP>:5000`

### Expand EBS Volume (if disk full)

```bash
sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1
```

### After EC2 Reboot

```bash
cd ~/gs_stock_prediction/deployment && docker compose up -d
```

Assign an **Elastic IP** or the public address changes on every reboot.

---

## MLflow Experiment Tracking

MLflow UI: `http://<EC2-PUBLIC-IP>:5000`

Each training run logs:

**Parameters tracked:**
- Model type and hyperparameters per model
- Split boundaries and row counts
- Feature count, selected ARIMA order
- Data validation summary (rows, date range, OHLC valid count)

**Metrics tracked:**
- RMSE, MAE, MAPE, SMAPE, R², Explained Variance
- Directional Accuracy, Information Coefficient
- Strategy Sharpe, Max Drawdown, Hit Rate
- **SKILL** (versus the naive baseline)
- QLIKE, QLIKE gain, MZ slope, MZ R², VaR breach, persistence
- ADF p-values, annualised volatility, excess kurtosis

**Artifacts stored:**
- `model_comparison.csv`, `volatility_comparison.csv`
- Feature importance (XGBoost)
- Forecast vs actual plot
- Registered XGBoost model with an inferred signature

### Experiment Structure

```
GS_Stock_Prediction/
├── pipeline_arima_prophet_xgboost_lstm/
│   ├── params: ticker=GS, arima_order=(2,1,3), n_features=35
│   └── metrics: xgboost_SKILL=0.0013, arima_SKILL=-0.0103,
│                naive_RMSE=8.8619, annualised_vol=0.359
└── volatility_garch/
    ├── params: target=conditional volatility, test_start=2023-01-01
    └── metrics: garch_QLIKE=-7.3945, QLIKE_gain=0.4127,
                 VaR99_Breach=0.93, persistence=0.9919
```

> **MLflow >= 3.0 note.** The plain-file backend (`file:///…/mlruns`) is in maintenance mode and MLflow refuses to start against it. `config.py` uses `sqlite:///mlruns/mlflow.db`, which needs no server for logging and enables the Model Registry.

---

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["gunicorn", "api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "180"]
```

### docker-compose.yml

| Service | Image / build | Port | Purpose |
|---|---|---|---|
| `gs-api` | built from `deployment/Dockerfile` | 8000 | FastAPI + Gunicorn |
| `gs-mlflow` | `ghcr.io/mlflow/mlflow` | 5000 | Tracking UI + Model Registry |

Both mount `../mlruns`, `../models_saved` and `../data` from the host, so training done inside a container persists after `docker compose down`.

---

## Testing

```bash
pytest tests/ -q      # 23 tests
```

| Test | Catches |
|---|---|
| `test_ohlc_invariants_hold` | corrupt or mislabelled source data |
| `test_split_is_chronological_and_disjoint` | shuffled splits |
| `test_features_contain_no_same_bar_information` | any feature correlating > 0.30 with the same bar's return |
| `test_target_aligns_with_previous_close` | the off-by-one that produced 98.8% accuracy |
| `test_leakage_detector_fires_on_perfect_predictions` | that the guard itself still works |
| `test_garch_beats_rolling_baseline_on_qlike` | that the headline volatility claim still holds |

---

## Results

### Price forecasting — one-step-ahead, test period 2023-01-03 to 2026-01-02

| Model | RMSE | MAE | MAPE | R² | Dir. acc. | **SKILL** | Training Time |
|---|---|---|---|---|---|---|---|
| XGBoost | **8.8501** | 5.9061 | 1.20% | 0.9972 | 54.98% | **+0.0013** | < 1s |
| Naive + drift | 8.8580 | 5.9125 | 1.20% | 0.9972 | 55.11% | +0.0004 | instant |
| **Naive (random walk)** | 8.8619 | 5.9171 | 1.21% | 0.9972 | — | 0.0000 | instant |
| ARIMA(2,1,3) | 8.9536 | 5.9531 | 1.22% | 0.9972 | 49.80% | −0.0103 | ~15s |
| Prophet | 137.7725 | 99.5205 | 17.56% | 0.3307 | 47.41% | −14.5466 | ~5s |

> **No model meaningfully beats the random walk.** XGBoost's +0.0013 is a 0.13% RMSE improvement from 35 engineered features, and early stopping halted after only 2 boosting rounds. Directional accuracy of 54.98% over 753 days is within sampling error of a coin flip.

**Reading the table.** R² = 0.997 and MAPE ≈ 1.2% are meaningless here — both come free from a series whose consecutive values are nearly identical, and the naive baseline scores the same. Report SKILL instead.

**AIC picked (2,1,3), and it still lost.** On the fitting window AIC ranks the pure random walk (0,1,0) **26th of 31** candidates, 20.9 AIC worse than the winner. In-sample the AR and MA terms improve the likelihood; out-of-sample they buy nothing.

**Prophet fails structurally, not through a bug.** It decomposes a series into trend plus yearly and weekly seasonality. Equity prices have no reliable seasonality, and Prophet never sees test observations — its trend is extrapolated across three years.

**Ignore the 55.11% for naive + drift.** Drift is positive, so that model always predicts "up" and the strategy is buy-and-hold. GS rose over the test window. That is market exposure, not forecasting skill.

### Volatility forecasting — one-step-ahead, same test period

| Model | QLIKE | QLIKE gain | MZ slope | MZ R² | Mean ann. vol | VaR99 breach |
|---|---|---|---|---|---|---|
| **GARCH(1,1) normal** | **−7.3945** | **+0.4127** | 1.6149 | **0.1610** | 26.96% | **0.93%** |
| GJR-GARCH(1,1,1)-t | −7.3930 | +0.4112 | 1.4206 | 0.1470 | 26.40% | 0.40% |
| GARCH(1,1)-t | −7.3898 | +0.4080 | 1.6061 | 0.1534 | 26.88% | 0.93% |
| EWMA (lambda 0.94) | −7.1400 | +0.1582 | 0.5357 | 0.0186 | 25.84% | 2.12% |
| Rolling 60-day | −7.1150 | +0.1332 | 0.3415 | 0.0048 | 26.03% | 2.12% |
| Rolling 20-day | −7.0332 | +0.0514 | 0.2882 | 0.0088 | 25.20% | 1.99% |
| Constant volatility | −6.9818 | 0.0000 | 0.0000 | 0.0000 | 36.84% | 0.53% |

> **This is the real result.** GARCH beats the constant-volatility baseline by 0.41 QLIKE — about 2.6x the gain from EWMA and 8x that of a 20-day rolling window. Compare the price table, where the best model beat its baseline by 0.0013.

**Why QLIKE and not RMSE.** True volatility is never observed, so any score uses a noisy proxy (here the squared return). Plain MSE on that proxy is dominated by a handful of large-return days. QLIKE is robust to proxy noise and penalises *under*-prediction of risk far more than over-prediction, which is the right asymmetry for a risk model.

**The VaR column is the practical test.** A calibrated 99% one-day VaR should be breached on ~1% of days. GARCH-normal breached on **0.93%**. EWMA and both rolling windows breached on ~2%, meaning a desk using them would carry twice the tail risk it believed. Constant volatility gets 0.53% by being wrong in the safe direction — its 36.84% average is a 26-year figure that overstates the calm 2023–2025 period.

**Persistence alpha + gamma/2 + beta ≈ 0.99** across all three specifications — volatility shocks decay slowly, which is volatility clustering stated numerically, and sitting just below 1 confirms the process is stationary rather than integrated.

**Best AIC is not best forecast.** GJR-GARCH-t fits far better in-sample (AIC 24,005 vs 24,381), exactly as excess kurtosis of 11.6 predicts. Out-of-sample it loses to plain GARCH, and its fatter tails make the 99% VaR too conservative at 0.40%.

### Why the split result

From `notebooks/01_EDA_Analysis.py`:

```
Return ACF lag 1  : -0.0456   <- near zero: direction is not predictable
|Return| ACF lag 1: +0.2883   <- large:     volatility IS predictable
```

Returns are unforecastable; their *magnitude* is strongly autocorrelated. That asymmetry is why the price pipeline fails and the volatility pipeline succeeds, and it is visible as volatility clustering around 2008 and 2020 in `reports/figures/01_overview.png`.

---

## Live Endpoints

Fill these in once deployed:

| Service | URL |
|---|---|
| FastAPI Swagger UI | `http://<EC2-PUBLIC-IP>:8000/docs` |
| FastAPI Health | `http://<EC2-PUBLIC-IP>:8000/health` |
| MLflow UI | `http://<EC2-PUBLIC-IP>:5000` |

---

## Disclaimer

Educational project. The results show that these models do not predict Goldman Sachs share prices better than assuming tomorrow's price equals today's. The volatility models do beat their baselines, but forecasting how much a stock will move is not forecasting which way it will move — a well-calibrated risk model is not a trading signal. Nothing here is investment advice.

---

## License

MIT License — feel free to fork and build on this.

---

## Author

Built and deployed end-to-end as a demonstration of a production ML system — from data validation and model training to cloud deployment with experiment tracking. Every figure in the Results section is reproduced by the committed pipelines and matches the committed CSVs in `data/processed/`.
