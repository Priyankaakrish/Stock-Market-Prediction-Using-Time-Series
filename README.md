📈 NVIDIA Stock Price Forecasting
=================================

End-to-end time-series ML pipeline for forecasting NVIDIA (NVDA) stock prices using **ARIMA, Prophet, XGBoost, and LSTM** — benchmarked against random-walk baselines, served via FastAPI, tracked with MLflow, and deployed on AWS EC2 with Docker Compose and Nginx.

`Python 3.12` · `FastAPI` · `MLflow 3` · `Docker` · `AWS` · `PyTorch` · `XGBoost` · `Prophet` · `PySpark`

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Finding](#-key-finding)
- [Architecture](#-architecture)
- [Data Flow](#-data-flow)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Feature Engineering](#-feature-engineering)
- [Leakage Controls](#-leakage-controls)
- [API Endpoints](#-api-endpoints)
- [Local Setup](#-local-setup)
- [AWS EC2 Deployment](#-aws-ec2-deployment)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Docker Configuration](#-docker-configuration)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Big Data Path (Spark / EMR)](#-big-data-path-spark--emr)
- [Results](#-results)
- [Limitations](#-limitations)
- [License](#-license)

---

## 🔍 Overview

This project builds a production-grade stock price forecasting system for NVIDIA (ticker: **NVDA**). It trains seven models — four ML/statistical models plus three baselines — exposes next-day predictions through a REST API, tracks every run in MLflow with a Model Registry, and is fully containerized for AWS EC2 deployment.

The system supports:

- Historical OHLCV ingestion with retries, validation, and offline fallback
- 46-feature causal, scale-free feature engineering pipeline
- Multi-model training (ARIMA · Prophet · XGBoost · LSTM) plus naive / MA / drift baselines
- **Statistical significance testing** (Diebold-Mariano) against a random-walk benchmark
- REST API for real-time forecasting with an empirical prediction interval
- MLflow UI + Model Registry for experiment tracking and model promotion
- Docker Compose orchestration behind an Nginx reverse proxy
- GitHub Actions CI/CD with a model quality gate and OIDC-based AWS deploy
- PySpark/EMR path for the multi-ticker / intraday scale-out

**Dataset:** 6,847 daily bars, 1999-01-22 → 2026-04-13. Zero missing values, zero duplicate dates.

---

## 🎯 Key Finding

> **No model in this project beats a random walk to any statistically significant degree.**

On a 1,019-day held-out window, the best model improves RMSE over "tomorrow's price equals today's price" by **0.08%**, with a Diebold-Mariano p-value of **0.62**. The apparent 53.8% directional accuracy is fully explained by the base rate of up-days in the test window (**53.78%**) — the winning model achieves it by always predicting *up*.

This is the correct result for daily equity closes, not a pipeline failure. Stock-prediction projects reporting sub-1% MAPE without a baseline row are usually measuring a lagged copy of the input — which is exactly what the `naive` row in the results table quantifies. The value of this repository is the harness that makes that visible: **baselines, a significance test, and a base-rate comparison, reported next to every headline number.**

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER                             │
│   • POST /forecast     • GET /health     • MLflow UI                │
└────────────────────┬───────────────────────┬────────────────────────┘
                     │ HTTPS :443            │ HTTPS :443
                     ▼                       ▼
              /  →  FastAPI            /mlflow/  →  MLflow UI
                     │                       │
┌────────────────────▼───────────────────────▼────────────────────────┐
│                     AWS EC2 (t3.large, Ubuntu 24.04)                │
│                        ap-south-1 (Mumbai)                          │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Docker Network (nvda-net)                    │  │
│  │                                                               │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │  │
│  │  │    nginx     │──▶│   nvda-api   │   │   nvda-mlflow    │  │  │
│  │  │  (80 / 443)  │   │  (FastAPI +  │   │ (MLflow Server + │  │  │
│  │  │              │──▶│   Gunicorn)  │   │  Model Registry) │  │  │
│  │  │  rate-limit  │   │   :8000      │   │      :5002       │  │  │
│  │  │  TLS term.   │   │              │──▶│                  │  │  │
│  │  └──────────────┘   │ • best_model │   │ SQLite backend   │  │  │
│  │   ONLY published    │ • /forecast  │   │ artifact store   │  │  │
│  │      ports          │ • /health    │   │                  │  │  │
│  │                     │ • /metrics   │   └────────┬─────────┘  │  │
│  │                     └──────────────┘            │            │  │
│  └─────────────────────────────────────────────────┼────────────┘  │
│                                                    │               │
│   ┌────────────────────────────────────────────────▼────────────┐  │
│   │              EBS Volume (20 GB, gp3)                         │  │
│   │   /mnt/mlflow → mlflow.db (metadata) + artifacts             │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │           AWS S3              │
                    │  raw/ · processed/ · mlflow-  │
                    │  artifacts/ (versioned, KMS)  │
                    └───────────────────────────────┘

AWS Security Group Inbound Rules:
  ┌──────────┬──────────┬─────────────┬──────────────────────────────┐
  │ Port     │ Protocol │ Source      │ Note                         │
  ├──────────┼──────────┼─────────────┼──────────────────────────────┤
  │ 443      │ TCP      │ 0.0.0.0/0   │ HTTPS — recommended          │
  │ 80       │ TCP      │ 0.0.0.0/0   │ redirect to 443              │
  │ 8000     │ TCP      │ —           │ CLOSED (internal only)       │
  │ 5002     │ TCP      │ —           │ CLOSED (proxied via /mlflow) │
  │ 22       │ TCP      │ —           │ CLOSED — deploys use SSM     │
  └──────────┴──────────┴─────────────┴──────────────────────────────┘
```

**Two deliberate differences from the common pattern:** only Nginx publishes ports — the API and MLflow stay on the internal bridge, so 8000 and 5002 are never exposed. And port 22 stays closed: deployments arrive over AWS SSM, so there is no SSH key material in CI.

---

## 🔄 Data Flow

```
NVIDIA OHLCV (Yahoo Finance / cached snapshot)
      │  6,847 rows · 1999-01-22 → 2026-04-13
      ▼
① Data Ingestion (yfinance)
      │  retries w/ backoff · schema validation · offline CSV fallback
      ▼
② Preprocessing
      │  dedupe · sort · forward-fill · zero-volume repair
      │  NO calendar reindex (weekends are not missing data)
      ▼
③ Feature Engineering  ──────────────────▶ 46 causal, scale-free features
      ├── MA ratios      close_over_MA7 / MA20 / MA50, MA7_over_MA50
      ├── Oscillators    RSI14, MACD_norm, MACD_signal, MACD_hist
      ├── Bollinger      BB_width, BB_pctB
      ├── Volatility     volatility_7 / _21, ATR14_norm, vol_ratio
      ├── Lags           log_ret_lag_{1,2,3,5,7,14,30}
      ├── Volume         vol_over_MA20, log_vol_change, dollar_vol_z
      └── Calendar       dow, month, is_month_end, is_quarter_end
      ▼
④ Chronological Split (70 / 15 / 15) — NO shuffling, no leakage
      │  train 4,750 │ val 1,018 │ test 1,019
      ▼
┌─────────────────────────────────────────────────────────┐
│                   Training Pipeline                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │  ARIMA   │ │ Prophet  │ │ XGBoost  │ │    LSTM    │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘ │
│  ┌────▼───────────-▼────────────▼─────────────▼──────┐  │
│  │   BASELINES: naive · ma5 · drift                  │  │
│  │   (the row most projects omit)                    │  │
│  └────────────────────┬──────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────┘
                        ▼
⑤ Evaluation — RMSE · MAE · MAPE · Directional Accuracy
                 · Diebold-Mariano test vs naive
                        ▼
⑥ MLflow Tracking (params · metrics · artifacts · 7 figures)
                        ▼
⑦ Model Registry — nvda-forecaster (pyfunc, versioned)
                        ▼
⑧ best_model.pkl + best_model_meta.json
                        ▼
⑨ FastAPI Service ──▶ /forecast  /health  /model  /metrics
```

---

## ✨ Features

- **Multi-model forecasting** — ARIMA, Prophet, XGBoost, LSTM in a single unified harness with one interface
- **Baselines that keep models honest** — naive (random walk), 5-day MA, and drift, scored identically
- **Statistical significance** — Diebold-Mariano test with Newey-West variance and Harvey small-sample correction
- **Return-based targets** — models learn log returns, not price levels (see [Models](#-models) for why this is decisive)
- **Leakage controls with tests** — 7 explicit assertions, because look-ahead bias never throws an exception
- **REST API** — FastAPI with Swagger UI at `/docs` and an empirical 80% prediction interval
- **Experiment tracking** — MLflow with SQLite backend and a real Model Registry entry (pyfunc, not a bare pickle)
- **Containerized** — multi-stage Docker build, non-root user, healthchecks, Compose orchestration
- **CI/CD** — GitHub Actions with lint → tests → smoke-train → quality gate → Trivy scan → ECR → SSM deploy
- **Big-data path** — PySpark job for EMR with a pandas-parity validation flag

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI + Gunicorn + Uvicorn workers |
| ML Models | statsmodels (ARIMA), Prophet, XGBoost, PyTorch (LSTM) |
| Baselines | Naive / Moving Average / Drift |
| Statistics | scipy (Diebold-Mariano), statsmodels (ACF diagnostics) |
| Experiment Tracking | MLflow 3.x (SQLite backend + Model Registry) |
| Data Source | yfinance (Yahoo Finance) + cached CSV snapshot |
| Feature Engineering | pandas, numpy (hand-rolled indicators — no ta-lib dependency) |
| Big Data | PySpark 3.5 on AWS EMR |
| Visualization | matplotlib |
| Testing | pytest (108 tests), ruff |
| Containerization | Docker (multi-stage) + Docker Compose |
| Web Server | Nginx (reverse proxy, TLS, rate limiting) |
| CI/CD | GitHub Actions + AWS OIDC + ECR + SSM |
| Cloud | AWS EC2 t3.large, Ubuntu 24.04, EBS 20 GB gp3, S3, CloudWatch |
| Region | ap-south-1 (Mumbai) |

---

## 📁 Project Structure

```
nvda-forecasting/
│
├── api/
│   ├── main.py                      # FastAPI app: /forecast /health /model /metrics
│   └── schemas.py                   # Pydantic request/response models
│
├── src/
│   ├── config.py                    # Paths + hyperparameters, single source of truth
│   ├── ingest.py                    # yfinance fetch, retries, validation, fallback
│   ├── preprocess.py                # Dedupe, sort, ffill, zero-volume repair
│   ├── features.py                  # 46 causal scale-free features
│   ├── split.py                     # Chronological split + walk-forward windows
│   ├── evaluate.py                  # RMSE/MAE/MAPE/DA + Diebold-Mariano
│   ├── train.py                     # Orchestrator: MLflow, selection, registry
│   ├── plots.py                     # 7 evaluation figures
│   ├── mlflow_model.py              # pyfunc wrapper for registry deployment
│   ├── models/
│   │   ├── base.py                  # Shared Forecaster interface
│   │   ├── baselines.py             # Naive / MovingAverage / Drift
│   │   ├── arima_model.py           # ARIMA + AIC grid search
│   │   ├── prophet_model.py         # Prophet w/ quarterly refit backtest
│   │   ├── xgb_model.py             # XGBoost on log returns
│   │   └── lstm_model.py            # PyTorch LSTM on log returns
│   └── spark_jobs/
│       └── prepare_data_spark.py    # PySpark/EMR feature engineering + parity check
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
│       └── s3_layout.md             # Data-lake layout + lifecycle policy
│
├── .github/workflows/ci-cd.yml      # Test → gate → build → scan → deploy
├── tests/test_pipeline.py           # 108 tests (leakage checks first)
│
├── data/raw/nvda_raw.csv            # Committed snapshot for offline CI
├── data/processed/                  # cleaned · features · train/val/test
├── models/                          # 7 pickles + best_model + metadata
├── reports/
│   ├── EVALUATION.md                # Full evaluation write-up
│   ├── metrics.csv                  # Model comparison table
│   ├── test_predictions.csv         # Per-day predictions, all models
│   └── figures/                     # 7 PNGs
│
├── Makefile                         # make data / train / test / api / mlflow
├── requirements.txt                 # Training environment
├── requirements-serve.txt           # Inference only (no torch/prophet)
└── requirements-dev.txt
```

---

## 🤖 Models

### 0. Baselines — naive · ma5 · drift

The row most stock-prediction projects omit. `naive` is `ŷ(t+1) = y(t)` — the random walk, and under weak-form market efficiency the theoretically optimal point forecast. Everything else is scored against it.

- **Speed:** instant
- **Why it matters:** a model reporting 2% MAPE that does not beat this has learned nothing

### 1. ARIMA (statsmodels)

Classical statistical model fitted on **`log(Close)`** with `d=1` — making it an ARMA on log returns, the stationary representation. Fitting on raw prices of a stock that grew 5,000x produces heteroskedastic residuals and unstable coefficients.

- **Params:** `order=(2,1,2)`, linear trend (a constant is annihilated by differencing); `grid_search_order()` selects by AIC
- **Backtest:** `append(refit=False)` — realised values are appended after each test day so every forecast is a true one-step-ahead with fixed coefficients, ~1000x faster than daily refits
- **Speed:** ~45s

### 2. Prophet (Meta)

Additive trend + seasonality decomposition. Included per the brief, and the result is informative: it shows what happens when you impose a smooth trend model on a near-martingale.

- **Params:** `changepoint_prior_scale=0.10`, weekly + yearly seasonality, `interval_width=0.80`
- **Backtest:** retrained every 63 trading days (~quarterly) and rolled forward — a realistic production cadence, not 1,000 Stan fits
- **Speed:** ~115s (16 refits)

### 3. XGBoost

Gradient-boosted trees on the 46 engineered features. **The target is `log(Close_t+1 / Close_t)`, not the price.**

> ⚠️ **This is the single most consequential decision in the project.** The chronological split puts the training maximum at **$6.23** while the test window reaches **$207.04**. Trees predict the mean of a leaf, so their output is bounded by the training target range. A price-level XGBoost would emit a flat line near $6 for the entire test period — RMSE ≈ $70 — while appearing to have trained successfully. Predicting returns removes the level entirely. `tests/test_pipeline.py::TestXGBoost::test_predicts_returns_not_levels` guards against anyone undoing this.

- **Params:** `n_estimators=600`, `max_depth=4`, `lr=0.02`, `subsample=0.8`, `min_child_weight=20`, early stopping on validation, then refit on train+val at the tuned tree count
- **Speed:** < 1s

### 4. LSTM (PyTorch)

Sequence model over a 30-day window of the same stationary features. Same target choice — a min-max scaler fitted on training *prices* has the mirror-image problem: every test price maps above 1.0, into a region the network never saw.

- **Architecture:** 2-layer LSTM (48 hidden) → Dropout → Dense(32) → Output
- **Params:** `lookback=30`, `dropout=0.2`, `lr=1e-3`, Huber loss, gradient clipping, early stopping (patience 6)
- **Serialization:** weights persisted as arrays and the graph rebuilt on load, so artifacts are portable across torch versions
- **Speed:** ~75s on CPU

---

## 🧪 Feature Engineering

46 model-facing features from 6,787 usable rows (60 dropped as indicator warm-up and label edge).

Two rules govern the module:

**1. Causality.** Every rolling statistic uses data up to and including day *t* only. No centred windows, no `shift(-1)` anywhere except when building the label. A test perturbs the final close and asserts no earlier indicator value changes.

**2. Stationarity.** `feature_columns()` filters out `Close`, `MA7`, `BB_upper`, `Volume_MA20` and every other non-stationary level, exposing only scale-free quantities. Levels are kept in the dataframe for plotting and price reconstruction, but never reach a model.

| Group | Features |
|---|---|
| Returns | `log_ret`, `ret_1d`, `ret_5d`, `ret_10d`, `ret_21d` |
| Volatility | `volatility_7`, `volatility_21`, `vol_ratio`, `ATR14_norm` |
| MA ratios | `close_over_MA7/20/50`, `MA7_over_MA50`, `MA20_over_MA50`, `ema12_over_ema26` |
| Oscillators | `RSI14`, `MACD_norm`, `MACD_signal_norm`, `MACD_hist_norm` |
| Bollinger | `BB_width`, `BB_pctB` |
| Candle shape | `hl_range`, `oc_change`, `close_pos_in_range`, `gap_open` |
| Volume | `vol_over_MA20`, `log_vol_change`, `dollar_vol_z` |
| Lags | `log_ret_lag_{1,2,3,5,7,14,30}`, `close_over_close_lag_{...}` |
| Calendar | `dow`, `month`, `is_month_end`, `is_quarter_end` |

---

## 🔒 Leakage Controls

Look-ahead bias in a price pipeline does not crash or throw — it produces a 99.9% R² and a model that loses money. The controls, each backed by a test:

| # | Control | Implementation |
|---|---|---|
| 1 | **Chronological splits only** | `chronological_split` asserts `train.max < val.min < test.max`. A random split is the most common error in this problem class. |
| 2 | **Causal indicators** | All rolling windows are trailing; a test perturbs the last close and asserts no earlier value moves |
| 3 | **Target isolation** | `target` / `target_return` are the only forward-looking columns and are excluded by name |
| 4 | **Correlation tripwire** | A test fails if any feature correlates with the target above \|0.99\| |
| 5 | **Honest backtests** | ARIMA appends realised values with fixed coefficients; Prophet retrains on a cadence and rolls forward — never fitted on the window it is scored against |
| 6 | **Scaler fitted on train only** | LSTM standardisation stats come from the training fold; validation windows prepend training rows for context |
| 7 | **No calendar reindex** | Forward-filling weekends would invent ~30% fake rows and flatten realised volatility |

---

## 🌐 API Endpoints

Base URL: `http://<EC2-PUBLIC-IP>/` (Nginx) or `http://localhost:8000` (local)

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "drift",
  "trained_at": "2026-09-08T13:53:08+00:00",
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

### `GET /metrics`

Prometheus text format — request counts, error counts, mean latency, model-loaded gauge. Restricted to private CIDR ranges in the Nginx config.

**Swagger UI:** `http://<EC2-PUBLIC-IP>/docs`

---

## 🔁 Running the offline pipeline

One command runs the whole offline band, timing each stage the architecture
draws as its own box:

```bash
python -m src.pipeline --mode local --fast --walk-forward --promote
```

```
ingest                  ok    0.3s  6847 rows, 1999-01-22 to 2026-04-13
preprocess              ok    0.1s  6847 rows, 100.0% weekdays (no calendar reindex)
feature engineering     ok    0.7s  6787 rows, 46/65 columns model-facing
train/val/test split    ok    0.6s  4750/1018/1019 rows, $6.23 -> $207.04 (33x), no leakage
train + evaluate        ok  141.3s  best=drift 2.9977 vs naive 3.0000; 3 of 5 beat naive
walk-forward validation ok   24.1s  16 folds; drift 10/16 p=0.454, xgboost 10/16 p=0.454
approval gate           ok    0.1s  approved
verify serving path     ok    0.2s  models:/nvda-forecaster@Production -> 188.8290 (+0.105%)
8 stages, 170.0s total
```

Three modes:

| Mode | Flow |
|---|---|
| `local` | ingest → pandas features → train. No AWS, no JVM. |
| `s3` | ingest → **land raw in S3** → features → **publish partitions** → train |
| `spark` | ingest → land raw → **EMR/Spark features** → publish → train |

Set `MODEL_URI=models:/nvda-forecaster@Production` to serve the approved
registry version instead of the baked-in pickle; both paths return identical
forecasts and there are tests asserting so.

Spark mode enforces the pandas parity check by default: swapping the feature
implementation under a trained model without proving agreement is how the two
paths silently diverge. S3 modes abort with a clear message when no bucket is
configured rather than skipping the stage.

The last stage matters most. Promotion moves a pointer; it does not prove the
thing pointed at can serve. One real inference through
`models:/nvda-forecaster@Production` runs on every promotion — the check that
would have caught the schema bug in `reports/EVALUATION.md` §9 on day one.

## 💻 Local Setup

### Prerequisites

- Python 3.12+
- Git
- (Optional) Docker + Docker Compose

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
pip install -r requirements-dev.txt

# 4. Build the dataset (ingest → clean → features → split)
make data

# 5. Train all seven models (~3 minutes)
make train
# Or CI mode: skips Prophet, shrinks the LSTM (~45s)
python -m src.train --fast

# 6. Run the test suite
make test                       # 108 tests
make lint                       # ruff

# 7. Start MLflow UI
make mlflow
# Open: http://localhost:5002

# 8. Start FastAPI
make api
# Open: http://localhost:8000/docs
```

### Verify

```bash
curl localhost:8000/health
curl -X POST localhost:8000/forecast \
  -H 'Content-Type: application/json' -d '{}'
```

---

## ☁️ AWS EC2 Deployment

### Infrastructure

| Setting | Value |
|---|---|
| Instance Type | t3.large (2 vCPU, 8 GB RAM) |
| OS | Ubuntu 24.04 LTS |
| Region | ap-south-1 (Mumbai) |
| Storage | Root 30 GB gp3 + EBS 20 GB gp3 for MLflow |
| Registry | Amazon ECR |
| Deploy channel | AWS SSM (no SSH) |

> **Why t3.large, not t3.micro:** 1 GB RAM cannot build a PyTorch image or train an LSTM without OOM-killing the container. If you only ever *serve* the model, `requirements-serve.txt` excludes torch and prophet entirely and a smaller instance works fine.

### Step 1 — Launch EC2 Instance

1. AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu 24.04 LTS**, instance type **t3.large**
3. Attach an **IAM instance profile** using `deploy/aws/iam_policy.json` (replace `REPLACE-BUCKET` and `REPLACE-ACCOUNT`)
4. Storage: 30 GB root + a 20 GB gp3 volume for MLflow
5. Security Group — inbound rules:

```
Port 443  → HTTPS  → 0.0.0.0/0
Port 80   → HTTP   → 0.0.0.0/0   (redirects to 443)
```

Note there is **no port 22 rule** and **no 8000/5002 rules**. Shell access goes through SSM Session Manager; the app and MLflow are reachable only through Nginx.

6. Assign an **Elastic IP** so the address survives reboots.

### Step 2 — Bootstrap the Instance

Paste `deploy/aws/ec2_bootstrap.sh` as user-data at launch, or run it after connecting. It installs Docker Engine + Compose, AWS CLI v2, the SSM agent, the CloudWatch agent, formats and mounts the EBS volume at `/mnt/mlflow`, and configures Docker log rotation.

```bash
sudo bash deploy/aws/ec2_bootstrap.sh
```

### Step 3 — Connect (no SSH keys)

```bash
aws ssm start-session --target i-0123456789abcdef0 --region ap-south-1
```

### Step 4 — Deploy the Application

```bash
sudo mkdir -p /opt/nvda-forecasting && cd /opt/nvda-forecasting
git clone https://github.com/<yourusername>/nvda-forecasting.git .

cp .env.example .env
nano .env                       # set S3_BUCKET, AWS_REGION, MODEL_URI

docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps
```

Expected:

```
NAME          STATUS             PORTS
nvda-nginx    running (healthy)  0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
nvda-api      running (healthy)
nvda-mlflow   running (healthy)
```

Only Nginx publishes ports — that is by design.

### Step 5 — Train on EC2 (or train locally and ship the artifact)

```bash
cd /opt/nvda-forecasting
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export MLFLOW_TRACKING_URI=http://localhost/mlflow
python -m src.train
```

### Step 6 — Enable TLS

```bash
sudo apt-get install -y certbot
sudo certbot certonly --standalone -d your.domain.com
sudo cp /etc/letsencrypt/live/your.domain.com/{fullchain,privkey}.pem \
        deploy/nginx/certs/
# Uncomment the 443 server block in deploy/nginx/nvda.conf, then:
docker compose -f deploy/docker-compose.yml restart nginx
```

### Step 7 — Verify

```bash
curl https://your.domain.com/health
# Swagger UI: https://your.domain.com/docs
# MLflow UI:  https://your.domain.com/mlflow/
```

### If the EBS volume fills up

```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h
docker image prune -af          # old ECR images are the usual culprit
```

### After a reboot

`restart: unless-stopped` on every service means the stack comes back on its own, and `nofail` in `/etc/fstab` keeps the EBS mount from blocking boot. With an Elastic IP assigned, the address does not change. No manual reattachment is required.

---

## 📊 MLflow Experiment Tracking

MLflow UI: `https://your.domain.com/mlflow/`

> **Note:** MLflow 3.x put the plain-file store into maintenance mode and raises on `file://` URIs. This project uses the **SQLite backend** (`sqlite:///mlflow.db`), which is also what the EBS-backed server uses in production. Point `MLFLOW_TRACKING_URI` at the remote server to log from anywhere.

Each pipeline run creates a parent run with nested child runs per model.

**Parameters tracked:** model type, all hyperparameters, ARIMA order, target representation (`log_return` vs `log_close`), feature count, split boundaries, train/test date ranges

**Metrics tracked:** `val_rmse`, `val_mae`, `val_mape`, `val_directional_accuracy`, `test_rmse`, `test_mae`, `test_mape`, `test_directional_accuracy`, `test_r2`, `fit_seconds`, `test_up_day_base_rate`, `best_skill_vs_naive_pct`, `naive_test_rmse`

**Artifacts stored:** trained model pickle, 7 figures, `metrics.csv`, `test_predictions.csv`, `feature_importance.csv`, split summary

**Tags:** `best_model`, `best_directional_model`, `beats_random_walk` — the caveat sits next to the result in the UI, not buried in a report.

### Experiment Structure

```
nvda-price-forecasting/
└── pipeline-20260908-141243/            (parent run)
    ├── naive     → test_rmse=3.000  mape=2.39%  DA=n/a
    ├── ma5       → test_rmse=4.230  mape=3.62%  DA=49.41%
    ├── drift     → test_rmse=2.998  mape=2.38%  DA=53.83%   ★ champion
    ├── arima     → test_rmse=4.084  mape=3.42%  DA=49.02%
    ├── prophet   → test_rmse=26.23  mape=27.21% DA=50.10%
    ├── xgboost   → test_rmse=3.001  mape=2.38%  DA=51.87%
    └── lstm      → test_rmse=2.998  mape=2.38%  DA=53.54%
```

### Model Registry

The champion is logged as a **pyfunc model**, not a bare pickle artifact — a bare pickle records the file but cannot be versioned or staged. Registered as `nvda-forecaster` with a signature, so serving can pull from the registry instead of a filesystem path:

```bash
export MODEL_URI=models:/nvda-forecaster/Production
```

---

## 🐳 Docker Configuration

### `deploy/Dockerfile.api`

Multi-stage build. Wheels compile once in the builder stage; the runtime stage carries no compiler.

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

> **The serving image excludes torch and prophet.** Shipping a 2 GB CUDA stack to serve a model that is a few hundred bytes of coefficients is waste. If an LSTM is promoted to champion, uncomment the CPU-only torch line in `requirements-serve.txt`.

### `deploy/docker-compose.yml`

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

Nginx routes `/` → `nvda-api:8000` and `/mlflow/` → `nvda-mlflow:5002`, applies a 30 req/min rate limit to `/forecast`, and restricts `/metrics` to private CIDRs.

---

## 🚀 CI/CD Pipeline

`.github/workflows/ci-cd.yml`:

```
push to main
    │
    ├── ① test              ruff → 39 pytest tests → smoke-train → upload metrics
    │
    ├── ② quality gate      train → block if best RMSE > 1.05 × naive
    │                              → block if best MAPE > 5%
    │
    ├── ③ build & push      buildx → ECR (sha + latest) → Trivy CVE scan
    │
    └── ④ deploy            SSM send-command → compose pull/up → health check ×12
```

**The quality gate is a regression gate, not a "must beat the market" gate.** Given the finding that nothing beats the random walk, a gate demanding market-beating skill would block every deploy forever. It instead blocks a champion that has become *materially worse* than the benchmark.

**Security:** GitHub authenticates to AWS via **OIDC** — no long-lived access keys in repository secrets. Deployment runs through **SSM**, so no SSH key material either.

---

## ⚡ Big Data Path (Spark / EMR)

The pandas path handles one ticker's 27-year history — 6,800 rows, comfortable on a laptop. `src/spark_jobs/prepare_data_spark.py` expresses the same logic in Spark for what the architecture is really sizing for: a universe of tickers, or intraday bars, where the feature table runs to hundreds of millions of rows.

```bash
# Local
spark-submit src/spark_jobs/prepare_data_spark.py \
    --input data/raw/nvda_raw.csv --output data/processed/spark

# EMR
spark-submit --deploy-mode cluster s3://<bucket>/jobs/prepare_data_spark.py \
    --input  s3://<bucket>/raw/nvda/ \
    --output s3://<bucket>/processed/nvda/ \
    --partition-by year
```

Both implementations are kept deliberately parallel, and `--validate-against` reads the pandas output and asserts the columns agree to 1e-6 — the only reliable way to stop two implementations drifting apart. S3 layout and lifecycle policy are documented in `deploy/aws/s3_layout.md`.

---

## 📈 Results

**Test window:** 2022-03-18 → 2026-04-10 (1,019 trading days). All forecasts are strict one-step-ahead.

### Splits

| Split | Rows | Start | End | Close min | Close max |
|---|---:|---|---|---:|---:|
| train | 4,750 | 1999-04-19 | 2018-03-02 | $0.03 | $6.23 |
| val | 1,018 | 2018-03-05 | 2022-03-17 | $3.18 | $33.38 |
| test | 1,019 | 2022-03-18 | 2026-04-10 | $11.23 | $207.04 |

> The **33× level gap** between the training maximum and the test maximum is the single most consequential fact in this project. It is what makes level-based modelling fail and forces the return-based target.

### Model Comparison

| Model | RMSE ($) | MAE ($) | MAPE (%) | Directional Acc. | Skill vs naive | DM p-value | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **drift** | **2.998** | **1.914** | **2.382** | 53.83% | +0.08% | 0.62 | < 1s |
| lstm | 2.998 | 1.914 | 2.384 | 53.54% | +0.06% | 0.74 | ~75s |
| naive | 3.000 | 1.919 | 2.386 | undefined | — | — | instant |
| xgboost | 3.001 | 1.916 | 2.383 | 51.87% | −0.04% | 0.91 | < 1s |
| arima | 4.084 | 2.705 | 3.423 | 49.02% | −36.1% | < 0.001 | ~45s |
| ma5 | 4.230 | 2.856 | 3.619 | 49.41% | −41.0% | < 0.001 | instant |
| prophet | 26.23 | 19.81 | 27.21 | 50.10% | −774% | < 0.001 | ~115s |

**Up-day base rate in the test window: 53.78%** — the number any directional accuracy must beat.

### Walk-forward validation (16 rolling quarters)

A single split gives one score per model; rolling-origin validation gives a
distribution, which is what separates a real edge from a lucky window.

| Model | RMSE mean | RMSE std | Folds beaten naive | Binomial p |
|---|---:|---:|---:|---:|
| drift | 2.513 | 1.675 | 10 / 16 | 0.454 |
| xgboost | 2.514 | 1.672 | 10 / 16 | 0.454 |
| naive | 2.515 | 1.673 | — | — |
| arima | 3.452 | 2.200 | 0 / 16 | <0.001 |
| ma5 | 3.575 | 2.280 | 0 / 16 | <0.001 |

Ten wins of sixteen; a coin flip gives eight, p = 0.454. Directional accuracy
swings ±5.8 pp fold to fold, twenty times the gap between drift's mean and the
base rate. ARIMA and MA5 lose in **every** fold — structure, not luck.

XGBoost's optimal tree count across the sixteen folds: 92, 208, 58, 198, 236,
181, 61, 15, 44, 44, 41, 43, 43, 42, 39, 14. A model whose optimal complexity
swings seventeen-fold between adjacent quarters is fitting noise.

### Drift detection

`python -m src.drift` compares test-window distributions against training via
PSI and a KS test. **4 of 46 features drifted significantly — all volatility or
range measures, all shifted downward.**

| Feature | PSI | Mean shift (sigma) |
|---|---:|---:|
| volatility_21 | 1.483 | −0.139 |
| ATR14_norm | 0.808 | −0.222 |
| log_vol_change | 0.271 | −0.004 |
| vol_over_MA20 | 0.262 | −0.041 |

Training spans the dot-com collapse and 2008; the test window does not. Momentum
and moving-average ratios are all stable — trend structure is unchanged, only
amplitude moved. This is what points at volatility as the target worth modelling
next.

### Model approval and promotion

`src/promote.py` is the "Best Approved Model" stage: the registry records
versions, this decides which one serves traffic.

```bash
python -m src.promote list       # every version and its alias
python -m src.promote check      # evaluate the newest against the gate
python -m src.promote promote    # set the Production alias
```

Three gates: not materially worse than the incumbent, MAPE under 5%, and — the
one this project exists to make — **not worse than a random walk**. A model that
loses to naive is more complex than the baseline and less accurate, so it is
blocked regardless of how it compares to the previous version.

### Walk-forward validation (16 rolling quarters)

The single split above gives one score per model. Rolling-origin validation
gives a distribution, which is what tells a real edge from a lucky window.

| Model | RMSE mean | RMSE std | Folds beaten naive | Binomial p |
|---|---:|---:|---:|---:|
| drift | 2.513 | 1.675 | 10 / 16 | 0.454 |
| xgboost | 2.515 | 1.676 | 9 / 16 | 0.804 |
| naive | 2.515 | 1.673 | — | — |
| arima | 3.451 | 2.199 | 0 / 16 | <0.001 |
| ma5 | 3.575 | 2.280 | 0 / 16 | <0.001 |

Drift wins 10 of 16 quarters; a coin flip gives 8, and p = 0.454. Its per-fold
skill swings from −0.59% to +0.77% with no persistence. ARIMA and MA5 lose in
**every** fold — structurally worse, not unlucky. Run it with
`python -m src.train --walk-forward --fast`.

### Reading the table

- **drift, lstm, naive, xgboost are one cluster.** Their RMSEs differ in the third decimal place; DM p-values above 0.6 say the differences are noise.
- **Naive has no directional accuracy.** It predicts zero change every day, so it never commits to a direction. Reporting 0% would mislead; the harness returns `NaN`.
- **drift wins because NVDA went up.** Its 53.83% directional accuracy is indistinguishable from the 53.78% base rate — it predicts "up" every single day. In a flat or falling regime it would be the worst performer in the table.
- **xgboost sits *below* the base rate** at 51.87%, meaning the feature-driven signal is slightly *worse* than always guessing up. Feature importances are diffuse with no dominant predictor — the signature of a model finding nothing.
- **arima and ma5 are significantly worse.** ARIMA's ARMA(2,2) terms fit autocorrelation that does not persist out of sample.
- **prophet fails informatively.** A 27% MAPE is what structural model mismatch looks like when a smooth trend-plus-seasonality decomposition meets a martingale.

### Figures

| File | Contents |
|---|---|
| `01_history_and_splits.png` | Log-scale price history with split boundaries + return series |
| `02_test_predictions.png` | All models vs actual over the test window, with error panel |
| `03_test_zoom.png` | Last 120 test days |
| `04_model_comparison.png` | RMSE / MAPE / directional accuracy with base-rate line |
| `05_feature_importance.png` | XGBoost gain-weighted importances |
| `06_residual_diagnostics.png` | Residual distribution, residual-vs-level, ACF |
| `07_lstm_learning_curve.png` | Train/validation loss |
| `08_walkforward.png` | Per-fold RMSE and skill across 16 rolling quarters |
| `08_walkforward.png` | Per-fold RMSE and skill-vs-naive across 16 rolling quarters |

Full analysis in [`reports/EVALUATION.md`](reports/EVALUATION.md).

---

## ⚠️ Limitations

- **Horizon is one day.** Multi-step forecasting needs recursive prediction with error accumulation, or direct multi-horizon models.
- **No transaction costs or slippage.** Directional accuracy is not a trading strategy — a 53.8% hit rate at retail spreads is not profitable.
- **Single asset, single regime.** The test window spans one exceptional bull run; these numbers would not transfer to a different asset or period.
- **Prophet's cadence is a choice, not a tuning result.** Quarterly refits are realistic; daily refits would score better at ~1,000 Stan fits per evaluation.
- **The Docker stack is validated locally** — multi-stage build, nginx routing, healthchecks and the internal `nvda-net` topology all confirmed running, with `/forecast` served through the proxy on port 80. The **AWS deployment path** (`ec2_bootstrap.sh`, SSM workflow, IAM policy) and the **PySpark job** remain unexecuted; both need infrastructure not available during development. The Spark job ships a `--validate-against` flag that asserts parity with the pandas implementation, which is the mechanism for verifying it on a cluster.
- **A 1999-2018 training window teaches the model about a company that no longer exists in the same form.** Training on a recent rolling window would fit the current regime at the cost of far less data.

### What would actually be worth trying

Ordered by expected value, based on what this evaluation ruled out:

1. **Change the target** — volatility clusters and *is* predictable at daily horizons; returns are not. A GARCH-family model would likely show real skill.
2. **Change the horizon** — weekly or monthly aggregation reduces microstructure noise, and momentum has some empirical support there.
3. **Change the features** — price and volume are the most-mined data in existence. Options-implied volatility, earnings surprises, or supply-chain data carry information not already in the price.
4. **Change the question** — "will tomorrow's move exceed ±2%" is a different and more tractable problem than predicting the price.

What is *not* worth trying: more technical indicators, deeper networks, or more hyperparameter search. The flat feature importances and the DM results say the signal is not there to be found.

---

## Live Resources

## Live Resources

| Service | Location |
|---|---|
| S3 data lake — raw | `s3://<bucket>/raw/nvda/ingest_date=YYYY-MM-DD/` |
| S3 data lake — processed | `s3://<bucket>/processed/nvda/year=YYYY/month=M/` |
| S3 — MLflow artifacts | `s3://<bucket>/mlflow-artifacts/` |
| FastAPI Swagger UI | `http://localhost:8000/docs` |
| FastAPI Health | `http://localhost:8000/health` |
| MLflow UI | `http://localhost:5002` |

> No public EC2 endpoint is listed because none is deployed. The S3 resources
> above are live in ap-south-1.
> 
## 📄 License

MIT License — feel free to fork and build on this.

## 👤 Author

Built end-to-end as a demonstration of a production ML system: data ingestion, causal feature engineering, multi-model training with honest baselines, statistical significance testing, experiment tracking, containerized serving, and cloud deployment with CI/CD.

**Stack:** Python · PyTorch · XGBoost · Prophet · statsmodels · FastAPI · MLflow · PySpark · Docker · Nginx · Gunicorn · AWS EC2/S3/ECR/SSM · GitHub Actions
