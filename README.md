# 📈 NVIDIA Stock Price Prediction

> End-to-end ML pipeline for forecasting NVIDIA (NVDA) stock prices using ARIMA, Prophet, XGBoost, and LSTM — served via FastAPI, tracked with MLflow, and deployed on AWS EC2 with Docker Compose.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.12.2-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Models](#models)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Configuration](#docker-configuration)
- [Results](#results)

---

## Overview

This project builds a production-grade stock price forecasting system for NVIDIA (ticker: `NVDA`). It trains four different time-series and ML models plus a weighted ensemble, exposes predictions through a REST API, tracks all experiments using MLflow, and is fully containerized and deployed on AWS EC2.

The system supports:
- Historical stock price ingestion (Jan 1999 – Sep 2026, 6,964 trading days) and feature engineering
- Multi-model training pipeline (ARIMA, Prophet, XGBoost, LSTM + Ensemble)
- REST API for real-time forecasting with configurable horizons
- MLflow UI for experiment tracking and model comparison
- Docker-based deployment on AWS EC2 (Mumbai region)

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
│                        AWS EC2 (t3.micro)                           │
│                       Ubuntu 26.04 LTS                              │
│                    ap-south-1b (Mumbai)                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Docker Network (nvda-net)                    │  │
│  │                                                               │  │
│  │   ┌─────────────────────────┐    ┌─────────────────────────┐  │  │
│  │   │        nvda-api         │    │       nvda-mlflow       │  │  │
│  │   │   (FastAPI + Uvicorn)   │    │    (MLflow 2.12.2)      │  │  │
│  │   │                         │    │                         │  │  │
│  │   │  • ARIMA model          │    │  Tracking UI :5000      │  │  │
│  │   │  • Prophet model        │    │  Experiment:            │  │  │
│  │   │  • XGBoost model        │    │  NVDA_Stock_Prediction  │  │  │
│  │   │  • LSTM model           │    │                         │  │  │
│  │   │  • Ensemble weights     │    │                         │  │  │
│  │   └────────────┬────────────┘    └────────────┬────────────┘  │  │
│  └────────────────┼──────────────────────────────┼───────────────┘  │
│                   │        bind-mounted volumes  │                  │
│  ┌────────────────▼──────────────────────────────▼───────────────┐  │
│  │                     EBS Volume (20 GB)                        │  │
│  │   models_saved/   data/   mlruns/   logs/                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

AWS Security Group Inbound Rules:
  ┌──────────┬──────────┬─────────────┐
  │ Port     │ Protocol │ Source      │
  ├──────────┼──────────┼─────────────┤
  │ 22       │ TCP      │ 0.0.0.0/0   │
  │ 8000     │ TCP      │ 0.0.0.0/0   │
  │ 5000     │ TCP      │ 0.0.0.0/0   │
  └──────────┴──────────┴─────────────┘
```

### Data Flow

```
Yahoo Finance API
      │
      ▼
Data Ingestion (yfinance → data/raw/nvda_*.csv)
      │
      ▼
Feature Engineering
  ├── Moving Averages (SMA/EMA 5, 10, 20, 50, 200)
  ├── RSI, MACD, Bollinger Bands, ATR, Stochastic
  ├── Lag Features & Rolling Stats
  ├── Volume Indicators (OBV, VWAP)
  └── Scale-free features (returns, ratios, z-scores)
      │
      ▼
┌──────────────────────────────────────────┐
│            Training Pipeline             │
│  ┌──────────────┐  ┌──────────────────┐  │
│  │    ARIMA     │  │     Prophet      │  │
│  └──────┬───────┘  └────────┬─────────┘  │
│  ┌──────▼───────┐  ┌────────▼─────────┐  │
│  │   XGBoost    │  │      LSTM        │  │
│  └──────┬───────┘  └────────┬─────────┘  │
│         └────────┬──────────┘            │
│          ┌───────▼────────┐              │
│          │    Ensemble    │              │
│          └───────┬────────┘              │
└──────────────────┼───────────────────────┘
                   ▼
           MLflow Tracking
     (metrics, params, artifacts)
                   │
                   ▼
           Saved Model Files
                   │
                   ▼
           FastAPI Service
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
  /api/v1/forecast        /api/v1/health
  /api/v1/models          /api/v1/predict/single
```

---

## Features

- **Multi-model forecasting** — ARIMA, Facebook Prophet, XGBoost, LSTM and a weighted ensemble in a single unified pipeline
- **REST API** — FastAPI with automatic Swagger UI at `/docs`
- **Experiment tracking** — MLflow UI with metrics, parameters, and artifacts logged per run
- **Containerized** — Docker Compose orchestrates API and MLflow services
- **Cloud deployed** — Running on AWS EC2 (Mumbai region)
- **Feature engineering** — 100+ technical indicators, lag features, rolling statistics, calendar features
- **Configurable forecast horizon** — Predict 1–252 business days ahead via API parameter
- **Honest evaluation** — One-step-ahead testing against a naive "tomorrow = today" baseline
- **Data refresh** — `pipelines/update_data.py` pulls the latest NVDA prices
- **Health checks** — API and model status endpoints

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **API Framework** | FastAPI + Uvicorn |
| **ML Models** | statsmodels / pmdarima (ARIMA), Prophet, XGBoost + Optuna, PyTorch (LSTM) |
| **Experiment Tracking** | MLflow 2.12.2 |
| **Data Source** | yfinance (Yahoo Finance) |
| **Feature Engineering** | pandas, numpy |
| **Containerization** | Docker + Docker Compose |
| **Cloud** | AWS EC2 t3.micro, Ubuntu 26.04, EBS 20GB |
| **Region** | ap-south-1b (Mumbai) |

---

## Project Structure

```
Stock-Market-Prediction-Using-Time-Series/
│
├── api/
│   ├── main.py                  # FastAPI app entry point (loads models at startup)
│   ├── schemas.py               # Pydantic request/response models
│   └── routers/
│       └── predict.py           # /api/v1 forecast, health, models, predict/single
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Load and merge NVDA CSVs
│   │   └── preprocessor.py      # Cleaning, splits, scalers
│   ├── features/
│   │   └── engineer.py          # Feature engineering
│   ├── models/
│   │   ├── arima_model.py       # ARIMA wrapper
│   │   ├── prophet_model.py     # Prophet wrapper
│   │   ├── xgboost_model.py     # XGBoost wrapper (Optuna tuning)
│   │   ├── lstm_model.py        # PyTorch BiLSTM + attention
│   │   └── ensemble.py          # Weighted ensemble
│   ├── evaluation/
│   │   └── metrics.py           # RMSE, MAE, MAPE, direction, Sharpe, IC
│   ├── visualization/
│   └── forecasting.py           # Shared one-step / multi-day forecasting logic
│
├── pipelines/
│   ├── update_data.py           # Download latest NVDA prices (yfinance)
│   ├── train_pipeline.py        # End-to-end training + MLflow logging
│   └── predict_pipeline.py      # Forward forecast from saved models
│
├── deployment/
│   ├── Dockerfile               # API container image
│   └── docker-compose.yml       # Multi-container orchestration
│
├── notebooks/
│   ├── 01_EDA_Analysis.py       # Exploratory data analysis
│   └── plots/                   # 8 EDA charts
│
├── tests/                       # pytest suite (data, models, API, forecasting)
├── mlruns/                      # MLflow experiment runs
├── models_saved/                # Saved model files
├── data/                        # Raw and processed data
├── logs/                        # Training logs
│
├── config.py                    # All settings (ticker, splits, hyperparameters)
├── requirements.txt
└── README.md
```

---

## Models

### 1. ARIMA
Classical statistical time-series model. Captures autoregressive and moving average components of the NVDA price, fitted on log price so NVIDIA's exponential growth becomes a linear drift.
- **Params:** order `(p, d, q)` selected via AIC with `auto_arima` → ARIMA(1,1,0) with drift
- **Use case:** Short-term trend forecasting
- **Speed:** Fast (seconds)

### 2. Facebook Prophet
Additive forecasting model by Meta. Handles seasonality, holidays, and trend changes automatically. Fitted on log price.
- **Params:** yearly/weekly seasonality, US holidays, changepoint prior scale 0.05
- **Use case:** Multi-period forecasting with seasonality
- **Speed:** Fast (seconds)

### 3. XGBoost
Gradient boosted trees on engineered time-series features. Predicts the next-day return, which lets it work across NVIDIA's whole price range ($0.04 → $227).
- **Params:** max_depth, learning rate, subsample, min_child_weight — tuned with Optuna (50 trials, 5-fold time-series CV)
- **Features:** 61 scale-free features — returns, lags, RSI, MACD, Bollinger %B, ATR ratio, z-scores
- **Use case:** Non-linear pattern capture
- **Speed:** ~1 minute (including hyperparameter search)

### 4. LSTM (PyTorch)
Deep learning sequence model. Learns long-term temporal dependencies in price sequences.
- **Architecture:** 2-layer Bidirectional LSTM (128, 64) → Self-Attention → Dense → Output
- **Params:** sequence_length=60, dropout=0.2, Huber loss, early stopping (patience 15)
- **Use case:** Complex temporal pattern learning
- **Speed:** ~1–2 min on a laptop CPU, slower on t3.micro

### 5. Ensemble
Weighted blend of the four models. Weights are optimised to minimise validation RMSE.
- **Latest weights:** Prophet 0.57, LSTM 0.20, ARIMA 0.12, XGBoost 0.11

---

## API Endpoints

Base URL: `http://13.233.91.41:8000`

### `GET /api/v1/health`
Returns API and model status.
```json
{
  "status": "healthy",
  "models_loaded": ["arima", "prophet", "lstm", "xgboost"],
  "last_data_date": "2026-09-29",
  "version": "1.0.0"
}
```

### `POST /api/v1/forecast`
Get stock price forecast.

**Request:**
```json
{
  "steps": 30,
  "models": ["ensemble"],
  "include_confidence_interval": true
}
```
`models` can be any of `arima`, `prophet`, `lstm`, `xgboost`, `ensemble`; `steps` is 1–252 business days.

**Response (shape):**
```json
{
  "ticker": "NVDA",
  "generated_at": "2026-09-30T08:24:13.866000Z",
  "forecast_days": 30,
  "models_used": ["ensemble"],
  "forecasts": {},
  "ensemble": [
    {"date": "2026-09-30", "predicted": 227.5, "lower_bound": 227.2, "upper_bound": 227.8},
    {"date": "2026-10-01", "predicted": 227.9, "lower_bound": 227.4, "upper_bound": 228.4}
  ],
  "metadata": {"last_known_date": "2026-09-29", "ci_included": true}
}
```

### `GET /api/v1/models`
Lists all loaded models and their parameters.

### `POST /api/v1/predict/single`
Predicted close N business days ahead (XGBoost).
```json
{"close": 227.21, "steps": 5}
```

**Swagger UI:** `http://13.233.91.41:8000/docs`

---

## Local Setup

### Prerequisites
- Python 3.10 – 3.12
- Git

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
pip install -r requirements.txt

# 4. Get the latest NVDA prices (optional — bundled data ends 2026-09-29)
python pipelines/update_data.py

# 5. Run training pipeline (all models, ~5 min on a laptop CPU)
python pipelines/train_pipeline.py
# or one model at a time:
python pipelines/train_pipeline.py --model arima
python pipelines/train_pipeline.py --model prophet
python pipelines/train_pipeline.py --model xgboost
python pipelines/train_pipeline.py --model lstm

# 6. Start MLflow UI locally
mlflow ui --backend-store-uri mlruns --port 5000
# Open: http://localhost:5000

# 7. Start FastAPI locally
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000/docs
```

> Windows: if `venv\Scripts\activate` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

---

## AWS EC2 Deployment

### Infrastructure

| Setting | Value |
|---|---|
| Instance Type | t3.micro (2 vCPU, 1GB RAM) + 2 GB swap |
| OS | Ubuntu 26.04 LTS |
| Region | ap-south-1b (Mumbai) |
| Storage | EBS 20 GB |
| Public IP | 13.233.91.41 |

### Step 1 — Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu** LTS
3. Select **t3.micro** (free tier eligible)
4. Create or select a key pair (`.pem` file)
5. Configure Security Group — add inbound rules:

```
Port 22    → SSH          → 0.0.0.0/0 (or My IP)
Port 8000  → FastAPI      → 0.0.0.0/0
Port 5000  → MLflow UI    → 0.0.0.0/0
```

6. Set storage to **20 GB** (default 8 GB is too small for Docker + PyTorch)
7. Launch instance

### Step 2 — Connect to the Instance

Use **EC2 Instance Connect** (AWS Console → Instances → select instance → **Connect**), or SSH:

```bash
# Windows (PowerShell)
ssh -i C:\Users\<you>\Downloads\<key>.pem ubuntu@<EC2-PUBLIC-IP>

# Linux/Mac
chmod 400 <key>.pem
ssh -i <key>.pem ubuntu@<EC2-PUBLIC-IP>
```

### Step 3 — Install Docker on EC2

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker

# Verify
sudo docker --version
sudo docker compose version
```

### Step 4 — Add Swap Memory (t3.micro)

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

### Step 5 — Get the Project

```bash
cd ~
git clone https://github.com/Priyankaakrish/Stock-Market-Prediction-Using-Time-Series.git
cd Stock-Market-Prediction-Using-Time-Series
```

### Step 6 — Build and Start Containers

```bash
cd deployment
sudo docker compose up -d --build

# Verify containers are running
sudo docker compose ps
```

Expected output:
```
NAME          STATUS    PORTS
nvda-api      running   0.0.0.0:8000->8000/tcp
nvda-mlflow   running   0.0.0.0:5000->5000/tcp
```

### Step 7 — Refresh Data and Retrain on EC2 (optional)

```bash
sudo docker compose exec nvda-api python pipelines/update_data.py
sudo docker compose exec nvda-api python pipelines/train_pipeline.py
sudo docker compose restart nvda-api
```

### Step 8 — Verify Deployment

```bash
# Test FastAPI
curl http://localhost:8000/api/v1/health

# Test from browser
# FastAPI Docs: http://<EC2-PUBLIC-IP>:8000/docs
# MLflow UI:    http://<EC2-PUBLIC-IP>:5000
```

### Expand EBS Volume (if disk full)

If you hit disk space errors during Docker build:

```bash
# After expanding volume in AWS Console (EC2 → Volumes → Modify volume):
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h /  # Verify new size
```

### After EC2 Reboot

The containers restart automatically (`restart: unless-stopped`). If needed:

```bash
cd ~/Stock-Market-Prediction-Using-Time-Series/deployment
sudo docker compose up -d
```

> **Tip:** Assign an AWS Elastic IP to your EC2 instance to keep a permanent public IP address.

---

## MLflow Experiment Tracking

MLflow UI is accessible at: `http://13.233.91.41:5000`

Each training run logs:

**Parameters tracked:**
- Ticker, target column, forecast horizon
- Train / validation / test date ranges
- ARIMA order, XGBoost Optuna best params, LSTM look-back / units / epochs
- Ensemble weights

**Metrics tracked (per model, on the test set):**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE / SMAPE (Mean Absolute Percentage Error)
- R², Directional Accuracy, Sharpe Ratio, Max Drawdown, IC

**Artifacts stored:**
- `model_comparison.csv` (test set)
- `model_comparison_val.csv` (validation set)
- `test_forecast.csv` (actual vs predicted, every model)

### Experiment Structure

```
NVDA_Stock_Prediction/
└── full_pipeline_all/
    ├── params: ticker=NVDA, arima_order=(1,1,0), lstm_lookback=60,
    │           test_period=2022-08-09..2026-09-29,
    │           ensemble_weights={prophet: 0.57, lstm: 0.20, arima: 0.12, xgboost: 0.11}
    └── metrics: ensemble_RMSE=3.38, arima_RMSE=3.38, lstm_RMSE=3.39,
                 prophet_RMSE=3.39, xgboost_RMSE=3.40, naive_last_close_RMSE=3.39
```

---

## Docker Configuration

### Dockerfile

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip && pip install torch==2.5.1 --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN mkdir -p models_saved logs data/raw data/processed mlruns
RUN useradd --uid 1000 --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

> CPU-only PyTorch is used to keep the image size manageable on t3.micro.

### docker-compose.yml

```yaml
services:
  nvda-api:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    image: nvda-stock-prediction:latest
    container_name: nvda-api
    ports:
      - "8000:8000"
    volumes:
      - ../models_saved:/app/models_saved
      - ../data:/app/data
      - ../mlruns:/app/mlruns
      - ../logs:/app/logs
    restart: unless-stopped
    networks:
      - nvda-net

  nvda-mlflow:
    image: python:3.11-slim
    container_name: nvda-mlflow
    command: >
      bash -c "pip install --no-cache-dir 'mlflow==2.12.2' 'setuptools<81' &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri /app/mlruns
               --default-artifact-root /app/mlruns"
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/app/mlruns
    restart: unless-stopped
    networks:
      - nvda-net

networks:
  nvda-net:
    driver: bridge
```

---

## Results

Test period **2022-08-09 → 2026-09-29** (1,081 trading days). Each model predicts the next day's close using data up to the previous day.

| Model | RMSE | MAE | MAPE | Directional Accuracy | Training Time |
|---|---|---|---|---|---|
| ARIMA | 3.384 | 2.214 | 2.18% | 50.7% | < 10s |
| Prophet | 3.389 | 2.227 | 2.17% | 48.9% | < 15s |
| XGBoost | 3.397 | 2.211 | 2.16% | 49.2% | ~1 min |
| LSTM | 3.387 | 2.205 | 2.16% | 51.8% | ~1–2 min (CPU) |
| **Ensemble** | **3.379** | 2.211 | 2.16% | 50.7% | — |
| Naive (tomorrow = today) | 3.386 | 2.201 | 2.16% | — | — |

> The Ensemble achieves the best RMSE on NVDA. All models stay within about 1% of the naive baseline — daily stock moves are close to unpredictable from price history alone, so this is the realistic benchmark.

---

## Live Endpoints

| Service | URL |
|---|---|
| FastAPI Swagger UI | http://13.233.91.41:8000/docs |
| FastAPI Health | http://13.233.91.41:8000/api/v1/health |
| MLflow UI | http://13.233.91.41:5000 |

---

## License

MIT License  — 

feel free to fork and build on this. 
---

