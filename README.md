# Stock-Market-Prediction-Using-Time-Series
# Goldman Sachs Stock Price Prediction

End-to-end time-series pipeline for Goldman Sachs (`GS`) using a naive baseline, ARIMA, Prophet, XGBoost and an LSTM — tracked with MLflow, served through FastAPI, containerised with Docker Compose, deployable on AWS EC2.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-3.0+-orange?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen)

---

## Table of contents

- [What this project claims](#what-this-project-claims)
- [Data integrity finding](#data-integrity-finding)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Volatility results](#volatility-results)
- [Models](#models)
- [API endpoints](#api-endpoints)
- [Local setup](#local-setup)
- [Docker](#docker)
- [AWS EC2 deployment](#aws-ec2-deployment)
- [MLflow tracking](#mlflow-tracking)
- [Testing](#testing)
- [Extensions](#extensions)

---

## What this project claims

It builds a correct, reproducible forecasting pipeline and reports what the models actually achieve.

**None of the models beat a random walk.** That is the finding, and it is reported in the results table rather than hidden. Any stock-prediction project reporting "1.9% MAPE" without showing that predicting *tomorrow = today* also scores ~1.2% has measured nothing — the metric is dominated by the fact that consecutive daily closes are nearly identical.

Every model is therefore scored on **SKILL**:

```
SKILL = 1 - RMSE_model / RMSE_naive
```

`SKILL > 0` beats the random walk. `SKILL ≤ 0` does not.

---

## Data integrity finding

The supplied `goldmansachs.csv` ships with **shifted column headers**. In 6,596 of 6,709 rows the `low` column exceeds the `high` column, which is impossible for a real price bar.

| Header in file | Actual field |
|---|---|
| `open` | Adjusted Close |
| `high` | Close |
| `low` | **High** |
| `close` | **Low** |
| `adj_close` | Open |

Two independent confirmations:

1. Under this mapping **all 6,709 rows** satisfy `High ≥ max(Open, Close)` and `Low ≤ min(Open, Close)`. No other permutation comes close.
2. `open ÷ high` rises smoothly from 0.692 (1999) to exactly 1.000 (2026) — the signature of a dividend-adjustment factor converging, which only makes sense if those two columns are Adj Close and Close.

`src/data/loader.py` applies the repair, then **re-validates and raises** if any row still fails. If you swap in a clean Yahoo Finance export, set `AUTO_REPAIR_COLUMNS = False` in `config.py`; the loader can also brute-force the correct permutation from the invariants alone.

Dataset after repair: **6,709 trading days, 1999-05-04 → 2026-01-02**, zero nulls, zero duplicate dates. The 250-day shortfall against the business-day calendar is exactly market holidays over 26 years.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      CLIENT / BROWSER                            │
└────────────────┬─────────────────────────┬───────────────────────┘
                 │                         │
      Port 8000 (FastAPI)         Port 5000 (MLflow UI)
                 │                         │
┌────────────────▼─────────────────────────▼───────────────────────┐
│                    AWS EC2 (t3.small)                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Docker network (gs-net)                       │  │
│  │  ┌──────────────────────┐   ┌───────────────────────────┐  │  │
│  │  │       gs-api         │   │        gs-mlflow          │  │  │
│  │  │  FastAPI + Gunicorn  │   │  mlflow server            │  │  │
│  │  │  • naive / drift     │   │  --host 0.0.0.0           │  │  │
│  │  │  • ARIMA             │   │  sqlite backend store     │  │  │
│  │  │  • Prophet           │   │                           │  │  │
│  │  │  • XGBoost           │   │  (binds all interfaces —  │  │  │
│  │  │  • LSTM              │   │   no TCP proxy needed)    │  │  │
│  │  └──────────────────────┘   └───────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  EBS 20 GB — ./mlruns/mlflow.db + ./mlruns/artifacts       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

Security group inbound:
  22   TCP  My IP        (SSH)
  8000 TCP  0.0.0.0/0    (FastAPI)
  5000 TCP  My IP        (MLflow — do not open to the world)
```

### Data flow

```
data/raw/goldmansachs.csv
      │
      ▼  src/data/loader.py       header repair + OHLC validation (raises on failure)
      ▼  src/data/preprocessor.py cleaning, chronological split, leak-free scaling
      ▼  src/features/engineer.py lags, RSI, MACD, Bollinger, ATR — all .shift(1)-ed
      │
      ├──► NAIVE baseline ◄── every model below is scored against this
      ├──► ARIMA        (AIC grid search, walk-forward one-step)
      ├──► Prophet      (trend + seasonality)
      ├──► XGBoost      (on log returns, early stopping on validation)
      └──► LSTM         (PyTorch, sequences of returns)
      │
      ▼  src/evaluation/metrics.py   RMSE · MAE · MAPE · DA · IC · Sharpe · SKILL
      ▼                              + automatic leakage detector
      ▼  MLflow (params, metrics, artifacts)
      ▼  FastAPI  /health · /api/v1/models · /api/v1/forecast
```

---

## Project structure

```
gs_stock_prediction/
├── config.py                     # every tunable parameter
├── requirements.txt
│
├── data/
│   ├── raw/goldmansachs.csv
│   └── processed/                # model_comparison.csv, test_forecast.csv
│
├── src/
│   ├── data/
│   │   ├── loader.py             # header repair + OHLC validation
│   │   └── preprocessor.py       # clean, temporal_split, DataScaler
│   ├── features/engineer.py      # indicators, lags, target alignment
│   ├── models/
│   │   ├── baseline.py           # naive + drift — the reference
│   │   ├── arima_model.py        # AIC grid search, walk-forward
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py         # PyTorch
│   │   └── garch_model.py        # GARCH/GJR + vol baselines
│   └── evaluation/metrics.py     # metrics + leakage detector
│
├── pipelines/
│   ├── train_pipeline.py         # price forecasting + MLflow logging
│   └── volatility_pipeline.py    # GARCH volatility forecasting
├── notebooks/01_EDA_Analysis.py
├── api/{main.py, schemas.py}
├── deployment/{Dockerfile, docker-compose.yml}
├── tests/{test_data.py, test_models.py, test_volatility.py, test_api.py}
└── reports/figures/
```

---

## Evaluation protocol

**Split** — chronological, never shuffled:

| Split | Rows | Period | Purpose |
|---|---|---|---|
| Train | 5,063 | 1999-05-04 → 2019-06-17 | fitting |
| Validation | 893 | 2019-06-18 → 2022-12-30 | tuning, early stopping |
| Test | 753 | 2023-01-03 → 2026-01-02 | held out, touched once |

After tuning, ARIMA and Prophet are refitted on train + validation. Skipping this leaves the model's last known price years stale, and the first test forecast is anchored to it — worth roughly 0.19 of spurious negative SKILL in this dataset.

**Forecast mode** — walk-forward one-step-ahead. Predict day *t* using only information through day *t−1*, then reveal the true value and move on. Parameters are estimated on training data and never refitted on the future; only the filter state advances. This is the only mode in which comparison against a naive baseline is meaningful.

**Target alignment** — every feature is `.shift(1)`-ed, and the target at row *t* is day *t*'s own return. Using `logret.shift(-1)` instead pairs row *t*'s features with the *t→t+1* return while the evaluator reads the actual price at *t*; the prediction gets compared against the wrong date and RMSE collapses toward zero. During development that bug produced **98.8% directional accuracy and SKILL 0.95** before it was caught. `check_for_leakage()` now fires automatically on any SKILL > 0.30 or directional accuracy > 65%, and `tests/test_data.py` asserts the alignment directly.

---

## Results

Test period 2023-01-03 → 2026-01-02 (753 trading days), one-step-ahead:

| Model | RMSE | MAE | MAPE | R² | Directional acc. | **SKILL** |
|---|---|---|---|---|---|---|
| XGBoost | 8.857 | 5.913 | 1.21% | 0.9972 | 54.1% | **+0.0006** |
| Naive + drift | 8.858 | 5.913 | 1.20% | 0.9972 | 55.1% | +0.0004 |
| **Naive (random walk)** | **8.862** | 5.917 | 1.21% | 0.9972 | — | 0.0000 |
| ARIMA(2,1,3) | 8.945 | 5.948 | 1.21% | 0.9972 | 49.7% | −0.0094 |
| Prophet | 138.170 | 99.736 | 17.58% | 0.3268 | 47.3% | −14.59 |

### Reading the table

**R² = 0.997 is meaningless here.** So is MAPE ≈ 1.2%. Both are what you get automatically from a series whose consecutive values are nearly identical. The naive baseline scores the same. Report SKILL instead.

**XGBoost's +0.0006 is noise, not skill.** Thirty-five engineered features across 5,000 training rows produced a 0.06% RMSE improvement over doing nothing. Directional accuracy of 54.1% over 753 days is within sampling error of a coin flip.

**The AIC grid search selected ARIMA(2,1,3), and it still lost to the random walk.** On the fitting window (2017-01 to 2022-12) AIC ranks the pure random walk (0,1,0) **26th of 31** candidates, 20.8 AIC worse than the winner. In-sample, the AR and MA terms genuinely do improve the likelihood. Out-of-sample they buy nothing: the selected model scores SKILL −0.0094, worse than predicting "tomorrow = today".

That gap is the point. AIC measures fit to data the model has already seen, and it will happily reward structure that does not generalise. It is the same lesson GJR-GARCH-t teaches in the volatility table below — best AIC, not best forecast — and it is why this project ranks models on SKILL against a naive baseline rather than on fit statistics.

**Prophet fails structurally, not through a bug.** It decomposes a series into trend plus yearly and weekly seasonality. Equity prices have no reliable seasonality, and Prophet never sees test observations — its trend is extrapolated from the end of training across three years. The 17.6% MAPE is what that assumption costs.

**Ignore the strategy Sharpe of 1.43 on naive+drift.** Drift is positive, so that model always predicts "up" and the strategy is buy-and-hold. GS rose over the test window. That is market exposure, not forecasting skill — which is also why "always predict up" scores 55.1% directional accuracy.

### What IS predictable

From `notebooks/01_EDA_Analysis.py`:

```
Return ACF lag 1  : -0.0456   <- near zero: direction is not predictable
|Return| ACF lag 1: +0.2883   <- large:     volatility IS predictable
```

Returns are unforecastable; their *magnitude* is strongly autocorrelated. This is the volatility clustering visible around 2008 and 2020. The volatility pipeline below acts on that.

---

## Volatility results

`pipelines/volatility_pipeline.py` — same split, same walk-forward discipline, same insistence on baselines. Test period 2023-01-03 → 2026-01-02, one-step-ahead:

| Model | QLIKE | QLIKE gain | MZ slope | MZ R² | Mean ann. vol | VaR99 breach |
|---|---|---|---|---|---|---|
| **GARCH(1,1) normal** | **−7.3945** | **+0.4127** | 1.615 | **0.161** | 27.0% | **0.93%** |
| GJR-GARCH(1,1,1)-t | −7.3930 | +0.4112 | 1.421 | 0.147 | 26.4% | 0.40% |
| GARCH(1,1)-t | −7.3898 | +0.4080 | 1.606 | 0.153 | 26.9% | 0.93% |
| EWMA (λ=0.94) | −7.1400 | +0.1582 | 0.536 | 0.019 | 25.8% | 2.12% |
| Rolling 60-day | −7.1150 | +0.1332 | 0.342 | 0.005 | 26.0% | 2.12% |
| Rolling 20-day | −7.0332 | +0.0514 | 0.288 | 0.009 | 25.2% | 1.99% |
| Constant volatility | −6.9818 | 0.0000 | 0.000 | 0.000 | 36.8% | 0.53% |

**This is a real result.** GARCH beats the constant-volatility baseline by 0.41 QLIKE — roughly 2.6× the gain from EWMA and 8× the gain from a 20-day rolling window. Contrast the price table above, where the best model beat its baseline by 0.0006.

**Why QLIKE and not RMSE.** True volatility is never observed, so any score uses a noisy proxy (here the squared return). Plain MSE on that proxy is dominated by a handful of large-return days. QLIKE is robust to proxy noise and penalises *under*-prediction of risk far more than over-prediction, which is the right asymmetry for a risk model.

**Mincer-Zarnowitz R² of 0.161 vs 0.019.** Regressing realised variance on forecast variance, GARCH explains ~16% of a proxy that is mostly noise; EWMA explains 2% and the rolling window under 1%. The slope of 1.61 shows GARCH's forecasts are under-dispersed — it compresses the range and would benefit from a scaling correction — but it is far closer to informative than the baselines, whose slopes near 0.3 mean their forecasts barely track realised variance at all.

**The VaR column is the practical test.** A calibrated 99% one-day VaR should be breached on ~1% of days. GARCH-normal breached on **0.93%** — near-perfect. EWMA and both rolling windows breached on ~2%, meaning a desk using them would take twice the tail risk it believed it was taking. Constant volatility gets 0.53% by being wrong in the safe direction: its 36.8% average annualised volatility is a 26-year average that badly overstates the calm 2023–2025 period.

**Persistence α + γ/2 + β ≈ 0.99** for all three specifications. Shocks to volatility decay slowly — the formal statement of "volatility clusters" — and the value sitting just below 1 confirms the process is stationary rather than integrated.

**GJR-GARCH-t has the best AIC but not the best QLIKE.** In-sample fit improves markedly with Student-t errors and a leverage term (AIC 24,005 vs 24,381), which is what the excess kurtosis of 11.6 predicts. Out-of-sample the three are within 0.005 QLIKE of each other, and the t-distribution's fatter tails make the 99% VaR too conservative (0.40% breach — over-reserving). Better in-sample fit did not transfer to better out-of-sample forecasts, which is worth a paragraph of its own in any writeup.

---

## Models

### Naive / drift — `src/models/baseline.py`
Random walk, optionally with drift. Not filler: it is the reference every other model is scored against, and it is reported first.

### ARIMA — `src/models/arima_model.py`
Grid search over (p,d,q) by AIC on log prices, fitted on the most recent 1,500 observations (ARIMA is short-memory; a 6,000-row fit costs minutes and adds nothing). Walk-forward via `.append(refit=False)`, which advances the state-space filter with new observations while holding parameters fixed. Selected order: **(2,1,3)**, AIC −7537.80.

### Prophet — `src/models/prophet_model.py`
Additive trend + seasonality on log prices. Included as a standard baseline and to demonstrate the structural mismatch.

### XGBoost — `src/models/xgboost_model.py`
400 trees, depth 4, lr 0.03, early stopping on the validation split (stopped at iteration 15). **Target is the next log return, not the price** — trees cannot extrapolate past their training range, and GS ended training near \$390 while ending testing near \$914, so a price-level model is capped on every prediction in the back half of the test set. Top features: `ret_mean5`, `vol_ratio`, `ret_std20`, `bb_pct`, `ret_mean50` — note that four of the five are *volatility* features, which is the model finding the only real signal in the data.

### GARCH family — `src/models/garch_model.py`
GARCH(1,1), GARCH(1,1)-t and GJR-GARCH(1,1,1)-t on log returns, with constant, rolling-window and RiskMetrics EWMA baselines. Parameters are estimated on the training window via `last_obs`; the variance recursion then runs forward on realised test returns, which is exactly the information a live model has each morning. This is the only pipeline here that beats its baseline.

### LSTM — `src/models/lstm_model.py`
Two-layer PyTorch LSTM (hidden 64, dropout 0.2, lookback 60) over sequences of scaled log returns, with early stopping. Scaler statistics come from the training block only; sequences are built first and split by date, never shuffled across the boundary.

---

## API endpoints

### `GET /health`
```json
{
  "status": "healthy",
  "data_loaded": true,
  "n_observations": 6709,
  "date_range": ["1999-05-04", "2026-01-02"],
  "models_available": ["naive", "arima"]
}
```

### `GET /api/v1/models`
Lists every model with its backtest RMSE and SKILL, so a caller can see the model does not beat a random walk before using it.

### `POST /api/v1/forecast`

Request:
```json
{ "model": "arima", "ticker": "GS", "horizon": 30 }
```

Response:
```json
{
  "ticker": "GS",
  "model": "arima",
  "horizon": 30,
  "last_observed_date": "2026-01-02",
  "last_observed_price": 914.34,
  "forecast": [
    { "date": "2026-01-05", "predicted_price": 914.34,
      "lower_bound": 878.21, "upper_bound": 951.95 }
  ],
  "backtest_metrics": { "RMSE": 8.9453, "MAPE": 1.2145, "SKILL": -0.0094 },
  "disclaimer": "Educational output …  Not investment advice."
}
```

Multi-step forecasts return confidence intervals where the model provides them. The ARIMA point forecast decays toward the drift within a few steps while the interval widens with roughly `√horizon`, so the horizon parameter should be read as "how uncertain", not "what price".

Swagger UI: `/docs`

---

## Local setup

```bash
git clone <your-repo-url> gs_stock_prediction
cd gs_stock_prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
# CPU-only torch first — the default PyPI wheel pulls ~2 GB of CUDA libraries
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1. Explore
python notebooks/01_EDA_Analysis.py

# 2. Train (baselines always run)
python pipelines/train_pipeline.py --model all
python pipelines/train_pipeline.py --model arima xgboost    # skip the slow ones

# 2b. Volatility — the pipeline that actually beats its baseline
python pipelines/volatility_pipeline.py

# 3. MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# 4. API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

> **MLflow ≥ 3.0 note.** The plain-file backend (`file:///…/mlruns`) is in maintenance mode and MLflow now refuses to start against it. `config.py` uses `sqlite:///mlruns/mlflow.db` instead — no server required, and the Model Registry works, which it never did on the file store.

---

## Docker

```bash
cd deployment
docker compose build
docker compose up -d
docker compose ps
```

| Container | Port | Purpose |
|---|---|---|
| `gs-api` | 8000 | FastAPI + Gunicorn |
| `gs-mlflow` | 5000 | MLflow UI |

The MLflow container binds `--host 0.0.0.0` and publishes port 5000 directly. No TCP proxy and no Nginx reverse proxy are needed — those are only necessary when the server binds to `127.0.0.1` inside the container.

---

## AWS EC2 deployment

**Use t3.small, not t3.micro.** 1 GB of RAM is not enough to `pip install` PyTorch and Prophet; the build gets OOM-killed. Set the EBS volume to **20 GB** — the 8 GB default fills during the Docker build.

```bash
# 1. Launch Ubuntu 24.04 LTS, t3.small, 20 GB EBS
#    Security group: 22 from My IP, 8000 from 0.0.0.0/0, 5000 from My IP

# 2. Connect
chmod 400 ubuntu-key.pem
ssh -i ubuntu-key.pem ubuntu@<EC2-PUBLIC-IP>

# 3. Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu && newgrp docker

# 4. Deploy
git clone <your-repo-url> ~/gs_stock_prediction
cd ~/gs_stock_prediction/deployment
docker compose up -d --build

# 5. Train inside the API container (models persist via the mounted volume)
docker compose exec gs-api python pipelines/train_pipeline.py --model arima xgboost

# 6. Verify
curl http://localhost:8000/health
```

Assign an **Elastic IP** or the public address changes on every reboot. Restrict port 5000 to your own IP — MLflow has no authentication and exposing it publishes your full experiment history.

If the disk fills mid-build:
```bash
sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1
```

---

## MLflow tracking

Each run logs:

- **Parameters** — model type, hyperparameters, split boundaries, row counts, feature count, selected ARIMA order
- **Metrics** — MAE, RMSE, MAPE, SMAPE, R², explained variance, directional accuracy, information coefficient, strategy Sharpe, max drawdown, hit rate, **SKILL** — per model, plus ADF p-values, annualised volatility and excess kurtosis for the run as a whole
- **Artifacts** — `model_comparison.csv`, `feature_importance.csv`, forecast-vs-actual plot
- **Tags** — a plain-language verdict, e.g. *"Best model: xgboost (SKILL +0.0006 vs naive)"*

---

## Testing

```bash
pytest tests/ -v      # 23 tests
```

The tests that matter:

| Test | Catches |
|---|---|
| `test_ohlc_invariants_hold` | corrupt or mislabelled source data |
| `test_split_is_chronological_and_disjoint` | shuffled splits |
| `test_features_contain_no_same_bar_information` | any feature correlating > 0.30 with the same bar's return |
| `test_target_aligns_with_previous_close` | the off-by-one that produced 98.8% accuracy |
| `test_leakage_detector_fires_on_perfect_predictions` | that the guard itself still works |
| `test_rolling_and_ewma_forecasts_use_no_same_day_data` | volatility forecasts peeking at the day they predict |
| `test_garch_beats_rolling_baseline_on_qlike` | that the headline volatility claim still holds |

---

## Extensions

Ranked by how likely they are to produce a result that survives scrutiny.

1. **Correct GARCH's under-dispersion.** The Mincer-Zarnowitz slope of 1.61 means forecasts are systematically compressed. Refit with a variance-targeting constraint, or apply the MZ regression as a post-hoc calibration, and re-check the VaR breach rate.
2. **HAR-RV or realised volatility from intraday data.** The squared daily return is an extremely noisy proxy; 5-minute realised variance would raise the achievable MZ R² well above 0.16.
3. **Direction classification at a longer horizon.** Predict the sign of the 5- or 10-day return; evaluate with a confusion matrix, ROC-AUC and a precision/recall trade-off rather than RMSE.
4. **Walk-forward cross-validation.** `TimeSeriesSplit` across several expanding windows, so one favourable test period cannot flatter the result.
5. **Exogenous features.** VIX, the 10-year/2-year spread, XLF sector returns, S&P 500 returns. Cross-sectional information is more likely to help than more transformations of the same price series.
6. **Transaction-cost-aware backtesting.** `strategy_metrics()` already charges 10 bps per turnover; sweep the cost and watch any apparent edge disappear.

---

## Disclaimer

Educational project. The results show that these models do not predict Goldman Sachs share prices better than assuming tomorrow's price equals today's. The volatility models do beat their baselines, but forecasting how much a stock will move is not forecasting which way it will move — a well-calibrated risk model is not a trading signal. Nothing here is investment advice, and none of it should be used to trade.

## License

MIT
