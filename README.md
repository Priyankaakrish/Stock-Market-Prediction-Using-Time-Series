# 📈 NVIDIA Stock Price Prediction

> End-to-end ML pipeline for forecasting NVIDIA (NVDA) stock prices with ARIMA, Prophet, XGBoost, a Bidirectional LSTM and a weighted ensemble — served via FastAPI, tracked with MLflow and containerised with Docker Compose.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.x-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-red?logo=pytorch)


## 📌 Table of Contents

- [Overview](#overview)
- [What changed for NVIDIA](#what-changed-for-nvidia)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [Docker & AWS EC2 Deployment](#docker--aws-ec2-deployment)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Testing](#testing)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

A stock-forecasting system for NVIDIA Corporation (ticker `NVDA`, NASDAQ), trained on daily prices from **Jan 1999 to Apr 2026** (6,847 trading days).

- Data loading, cleaning and 100+ engineered features (technical indicators, lags, rolling stats, calendar, scale-free ratios)
- Four models plus an ensemble whose weights are optimised on a validation period
- Honest one-step-ahead evaluation against a naive "tomorrow = today" baseline
- 1–252 business-day forecasts through a REST API and a batch script
- MLflow tracking of parameters, metrics and artefacts
- Docker Compose deployment (API + MLflow, optional Prometheus/Grafana)

---

## What changed for NVIDIA

NVIDIA's split-adjusted price grew from about **$0.04 to $190** — a ~5,000× rise, most of it after 2016. With a 70 / 15 / 15 time split, every training price is at or below about $6.2 while the test period runs from about $11 to $207. The Goldman Sachs design worked on price levels, which breaks on a series like this, so the following was changed:

| Area | Goldman Sachs version | NVIDIA version | Why |
|---|---|---|---|
| Outlier handling | IQR clamp on raw prices | IQR on each value's log-distance from its 21-day rolling median | The level-based clamp flattened ~920 days of the real AI rally to a single price |
| XGBoost & LSTM target | Price level | Next-day log return; price rebuilt as `close × exp(return)` | Trees can't predict above their training maximum; LSTM min-max scaling broke 30× out of range |
| XGBoost & LSTM inputs | All features, incl. same-day price levels | 61 scale-free features (returns, ratios, oscillators, z-scores) | Removes same-day leakage and keeps inputs in the same range at $1 and $190 |
| ARIMA & Prophet | Raw price | log(price) | Turns exponential growth into linear drift |
| Evaluation | Multi-step forecasts compared with misaligned dates | Every model one step ahead (uses data up to day t-1 only), plus a naive baseline | Fair, like-for-like comparison |
| Live forecasts | ARIMA/Prophet forecast from the end of the *training* period; XGBoost "rolled" its feature vector | ARIMA/Prophet refitted on full history; XGBoost/LSTM forecast recursively with features recomputed each day | Forecasts now start from the latest close |
| API | `/forecast`, `/models`, `/predict/single` crashed (`request=None`) | Fixed and covered by tests | — |
| Loader | De-duplication mask applied to re-sorted rows | Fixed | Duplicate dates slipped through |
| Data refresh | Static CSVs | `pipelines/update_data.py` downloads the latest NVDA prices | — |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       CLIENT / BROWSER                           │
└───────────────────┬─────────────────────────┬────────────────────┘
                    ▼                         ▼
          Port 8000 (FastAPI)        Port 5000 (MLflow UI)
┌───────────────────┴─────────────────────────┴────────────────────┐
│                 Docker Compose (network: nvda-net)               │
│   ┌─────────────────────────┐     ┌──────────────────────────┐   │
│   │ nvda-api (FastAPI)      │     │ nvda-mlflow              │   │
│   │  • ARIMA   • Prophet    │     │  experiment tracking     │   │
│   │  • XGBoost • LSTM       │     │  (mlruns/ volume)        │   │
│   │  • Ensemble weights     │     └──────────────────────────┘   │
│   └─────────────────────────┘                                    │
│   optional profile "monitoring": Prometheus :9090, Grafana :3000 │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow

```
Yahoo Finance ──► pipelines/update_data.py ──► data/raw/nvda_*.csv
                                                    │
                    loader (merge, de-duplicate) ◄──┘
                                │
                preprocessor (business-day calendar, bad-tick removal, returns)
                                │
             feature engineering (indicators, lags, rolling, calendar, scale-free)
                                │
          ┌─────────────┬───────┴───────┬──────────────┐
        ARIMA        Prophet         XGBoost          LSTM
      (log price)  (log price)   (next-day return) (next-day return)
          └─────────────┴───────┬───────┴──────────────┘
                    Ensemble (weights fitted on validation)
                                │
          MLflow (params, metrics, artefacts)  +  models_saved/
                                │
                  FastAPI  /  predict_pipeline.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **API** | FastAPI + Uvicorn |
| **Models** | statsmodels / pmdarima (ARIMA), Prophet, XGBoost + Optuna, PyTorch (Bi-LSTM + attention) |
| **Experiment tracking** | MLflow |
| **Data** | pandas, numpy, yfinance |
| **Containers** | Docker + Docker Compose |
| **Testing** | pytest, FastAPI TestClient |

---

## Project Structure

```
nvda_stock_prediction/
├── api/
│   ├── main.py                  # app factory, loads models at startup
│   ├── schemas.py               # Pydantic request/response models
│   └── routers/predict.py       # /api/v1 endpoints
├── config.py                    # all settings (ticker, splits, hyper-parameters)
├── data/
│   ├── raw/                     # nvda_master_dataset.csv, nvda_yahoo_finance.csv
│   └── processed/               # model_comparison*.csv, test_forecast.csv, forecast_output.csv
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models_saved/                # trained NVDA models (ready for the API)
├── notebooks/
│   ├── 01_EDA_Analysis.py       # exploratory analysis script
│   └── plots/                   # 8 EDA charts
├── pipelines/
│   ├── update_data.py           # download latest NVDA prices
│   ├── train_pipeline.py        # train + evaluate + log to MLflow
│   └── predict_pipeline.py      # forward forecast from saved models
├── src/
│   ├── data/                    # loader.py, preprocessor.py
│   ├── features/engineer.py     # feature engineering
│   ├── forecasting.py           # shared one-step / recursive forecasting logic
│   ├── models/                  # arima, prophet, xgboost, lstm, ensemble
│   └── evaluation/metrics.py    # MAE, RMSE, MAPE, direction, Sharpe, IC …
├── tests/                       # 60 tests
├── install_windows.bat
└── requirements.txt
```

---

## Models

### 1. ARIMA
- `auto_arima` order search on the last 500 days of **log price** → selected **ARIMA(1,1,0) with drift**
- Evaluated one step ahead with parameters fixed from training; production model refitted on recent history

### 2. Facebook Prophet
- Fitted on **log price** with yearly + weekly seasonality and US holidays
- Next-day prediction = last close × exp(Prophet's expected change for the next day)

### 3. XGBoost
- Predicts the **next-day log return** from 61 scale-free features
- Hyper-parameters tuned with **Optuna** (50 trials, 5-fold time-series CV), early stopping on validation

### 4. Bidirectional LSTM (PyTorch)
- 2 Bi-LSTM layers (128, 64) + self-attention, 60-day look-back window
- Predicts the **next-day log return** (in percent); Huber loss, early stopping

### 5. Ensemble
- Weighted blend; weights minimise validation RMSE → Prophet 0.61, XGBoost 0.28, ARIMA 0.06, LSTM 0.05

---

## Results

Chronological split: **train** 1999-02 → 2018-02, **validation** 2018-02 → 2022-03, **test 2022-03-17 → 2026-04-13** (1,063 trading days; NVDA traded between $11 and $207).
All numbers are one-step-ahead (predict tomorrow's close using data up to today). Full table: `data/processed/model_comparison.csv`.

| Model | MAE ($) | RMSE ($) | MAPE (%) | Direction accuracy (%) | Long/flat Sharpe |
|---|---|---|---|---|---|
| ARIMA | 1.848 | 2.947 | 2.32 | 51.0 | 1.11 |
| Prophet | 1.856 | 2.933 | 2.31 | 49.0 | 1.17 |
| LSTM | 1.843 | 2.941 | 2.30 | **51.3** | 1.11 |
| XGBoost | 1.861 | 2.967 | 2.32 | 48.7 | 0.90 |
| **Ensemble** | 1.847 | **2.931** | 2.30 | 50.4 | **1.29** |
| Naive (tomorrow = today) | **1.841** | 2.938 | **2.29** | — | — |
| Buy & hold NVDA | — | — | — | — | 1.20 |

**How to read this:**
- Every model tracks the price closely (R² ≈ 0.998), but so does simply repeating today's close. The fair benchmark is the naive row, and all models are within about 1% of it.
- The ensemble has the lowest RMSE and the best Sharpe of the strategies, but the edge over naive / buy-and-hold is small and not statistically established.
- Direction accuracy near 50% means daily moves are essentially unpredictable from past prices alone — the expected result for a heavily traded stock.

These are realistic numbers. Much lower errors in stock-prediction projects usually come from look-ahead leakage or from not comparing against the naive baseline.

---

## API Endpoints

Base URL (local): `http://localhost:8000` · Swagger UI: `http://localhost:8000/docs`

### `GET /api/v1/health`
```json
{"status": "healthy", "models_loaded": ["arima", "prophet", "lstm", "xgboost"],
 "last_data_date": "2026-04-13", "version": "1.0.0"}
```

### `POST /api/v1/forecast`
```json
{"steps": 30, "models": ["ensemble"], "include_confidence_interval": true}
```
`models` can be any of `arima`, `prophet`, `lstm`, `xgboost`, `ensemble`; `steps` is 1–252 business days. Returns dated forecasts per model and for the ensemble (with a spread-based band).

### `GET /api/v1/models`
Loaded models and their parameters (ARIMA order/AIC, XGBoost Optuna params).

### `POST /api/v1/predict/single`
```json
{"close": 189.31, "steps": 5}
```
Predicted close `steps` business days after the latest data (XGBoost).

---

## Local Setup

### Prerequisites
- Python 3.10 or 3.11
- ~3 GB disk for dependencies (PyTorch, Prophet)

### Steps

```bash
# 1. Get the code
git clone https://github.com/<your-username>/nvda_stock_prediction.git
cd nvda_stock_prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies (CPU PyTorch keeps it small)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. (Optional) download the latest NVDA prices — bundled data ends 2026-04-13
python pipelines/update_data.py

# 5. Train all models (~5 min on a laptop CPU)
python pipelines/train_pipeline.py            # or --model arima|prophet|lstm|xgboost

# 6. Forecast the next 30 business days
python pipelines/predict_pipeline.py --steps 30

# 7. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs

# 8. MLflow UI
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://localhost:5000

# EDA charts -> notebooks/plots/
python notebooks/01_EDA_Analysis.py
```

Trained models are already included in `models_saved/`, so step 7 works without retraining.

---

## Docker & AWS EC2 Deployment

```bash
cd deployment
docker compose up -d --build            # API on :8000, MLflow on :5000
docker compose --profile monitoring up -d   # + Prometheus :9090, Grafana :3000
docker compose ps
curl http://localhost:8000/api/v1/health
```

**On AWS EC2** (e.g. Ubuntu 24.04, t3.small or larger — PyTorch + Prophet are tight on 1 GB RAM):

1. Launch the instance with a 20 GB EBS volume; open inbound ports 22 (your IP), 8000 and 5000 in the security group.
2. SSH in and install Docker: `sudo apt update && sudo apt install -y docker.io docker-compose-v2`.
3. Copy the project up: `scp -i key.pem -r nvda_stock_prediction ubuntu@<EC2-IP>:~`.
4. `cd ~/nvda_stock_prediction/deployment && sudo docker compose up -d --build`.
5. Check `http://<EC2-IP>:8000/docs` and `http://<EC2-IP>:5000`.

Tip: attach an Elastic IP so the address survives reboots.

---

## MLflow Experiment Tracking

Experiment: **`NVDA_Stock_Prediction`**. Each training run logs:

- **Parameters:** ticker, train/val/test periods, ARIMA order, XGBoost Optuna params, LSTM look-back/units/epochs, ensemble weights
- **Metrics (test set, per model):** MAE, RMSE, MAPE, SMAPE, R², direction accuracy, Sharpe, max drawdown, IC
- **Artefacts:** `model_comparison.csv`, `model_comparison_val.csv`, `test_forecast.csv`

---

## Testing

```bash
pytest -q          # 58 passed, 2 skipped
```

Covers the loader, preprocessing, features, every model wrapper, all API endpoints and the NVDA-specific logic in `tests/test_forecasting.py` (trend-safe outlier handling, scale-free features, no look-ahead alignment, recursive forecasting).

---

## Limitations

- Uses only NVIDIA's own price and volume history — no earnings, news, macro or sector data, which drive most large moves.
- Recursive multi-day forecasts carry forward recent drift; uncertainty grows quickly beyond a few days and the ensemble band (model spread) understates it.
- Data refresh needs internet access to Yahoo Finance.

## Disclaimer

For education and portfolio use only. Not financial advice.

---

## License

MIT License
