# Goldman Sachs Stock Price Prediction — Time Series

ARIMA, Prophet, XGBoost, LSTM and a weighted ensemble, benchmarked against a
random walk, with a FastAPI service and Docker deployment.
with the evaluation methodology tightened so the numbers mean something.

Data: `goldmansachs.csv`, 6,709 rows, 1999-05-04 to 2026-01-02, after
repairing a column-ordering fault in the source file.

---

## Headline result

**No model beats a random walk with drift by a meaningful margin.** The best
ensemble improves RMSE by 3.5% over a two-line baseline with zero parameters.
Reported directional accuracy of ~62% is the test window's upward drift, not
forecasting skill.

That is the normal finding for daily equity prices. The sections below are
arranged so it stays visible rather than getting buried under a good-looking
RMSE.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. repair the source CSV (see Data repair)
python pipelines/repair_csv.py --in data/raw/goldmansachs.csv \
                               --out data/raw/goldmansachs_clean.csv

# 2. exploratory analysis → notebooks/plots/
python notebooks/01_EDA_Analysis.py --csv data/raw/goldmansachs_clean.csv

# 3. fit ensemble weights on validation, then train + evaluate
python pipelines/fit_ensemble.py   --csv data/raw/goldmansachs_clean.csv
python pipelines/train_pipeline.py --csv data/raw/goldmansachs_clean.csv

# 4. forecast from saved artefacts
python pipelines/predict_pipeline.py --steps 30

# 5. serve
uvicorn api.main:app --reload --port 8000   # docs at /docs

pytest tests/ -q                             # 49 tests
```

Or via `make`:

```bash
make install
make all        # repair -> eda -> ensemble -> train
make serve
make test
```

Models can be trained one at a time; predictions are cached per model in
`data/processed/cache/`, so a run picks up where the last one stopped.

```bash
python pipelines/train_pipeline.py --models arima
python pipelines/train_pipeline.py --report-only
```

Docker:

```bash
cd deployment
docker compose run --rm trainer     # repair → weights → train
docker compose up -d api            # http://localhost:8000/docs
```

---

## Data repair

The supplied CSV's header did not match its contents. The header reads
`date,open,high,low,close,adj_close,volume`, but on the first row `low`
(77.25) exceeds `high` (70.375), which is impossible.

Testing all 120 column permutations against the OHLC invariants
(`high >= max(open, close)`, `low <= min(open, close)`) left two candidates at
100% validity, differing only in whether open and close were swapped. Mean
overnight gap settles it — 0.72% for one ordering against 1.98% for the other,
since real markets open near the previous close.

**The five price columns are stored in exact reverse of their header labels.**
True order: `date, adj_close, close, high, low, open, volume`.

Two independent checks confirm it. The `adj_close / close` ratio runs 0.6920
to 1.0000 and converges on 1.0 at the most recent date, which is what an
adjusted-close series must do. And the 1999-05-04 adjusted close under this
reading is 48.697, against 48.44 in the reference yfinance dataset for the
same day.

`pipelines/repair_csv.py` detects this rather than hardcoding it, and puts
OHLC on a consistent adjusted basis (raw OHLC scaled by `adj_close / close`).
After repair: 0 invalid OHLC rows out of 6,709, no duplicate dates, no nulls.

Had this gone unnoticed, every model would have trained on a series where
"high" was actually the daily low.

---

## Evaluation design

Chronological 70/15/15 split — train 1999-2018 (4,871 days), validation
2018-2022 (1,044), test 2022-01-04 to 2026-01-02 (1,044). No shuffling.

Two tracks, both keyed by **target date** (the day being predicted):

- **Track A — 1 day ahead.** Walk-forward: predict tomorrow, then see the true
  value before moving on.
- **Track B — 30 business days ahead.** A true 30-day-ahead point forecast from
  every test date. This is the horizon the API serves.

Three reference points are reported as first-class models:

| Baseline | Definition |
|---|---|
| **Naive** | Tomorrow's price equals today's price |
| **Drift** | Today's price grown at the mean historical rate (fitted on train only) |
| **Theil's U** | Model RMSE ÷ naive RMSE. Below 1.0 beats the random walk |

Prophet appears only in Track B: it has no autoregressive term, so grading it
one-day-ahead would be meaningless rather than merely unflattering.

Ensemble weights are fitted on the **validation** period by
`pipelines/fit_ensemble.py`, using a second independent fit of every model on
the training split alone, then frozen before the test period is touched.

---

## Results

### Track A — 1 day ahead (n = 1,044)

| Model | RMSE | MAE | MAPE % | DirAcc % | IC | Theil U |
|---|---|---|---|---|---|---|
| XGBoost | 7.984 | 5.277 | 1.206 | 52.9 | 0.053 | 0.998 |
| LSTM | 7.985 | 5.284 | 1.209 | 53.0 | −0.049 | 0.998 |
| **Drift** | 7.994 | 5.283 | 1.207 | 53.0 | −0.033 | 0.999 |
| **Naive** | 8.004 | 5.287 | 1.207 | — | — | 1.000 |
| ARIMA | 8.062 | 5.390 | 1.234 | 47.5 | −0.015 | 1.007 |

Spread across all five: 1.0%. ARIMA is worse than doing nothing.

### Track B — 30 business days ahead (n = 1,014)

| Model | RMSE | MAE | MAPE % | DirAcc % | IC | Theil U |
|---|---|---|---|---|---|---|
| **Ensemble** | 43.47 | 33.44 | 7.63 | 63.5 | 0.165 | 0.934 |
| LSTM | 43.75 | 33.18 | 7.49 | 61.8 | 0.068 | 0.940 |
| XGBoost | 44.89 | 33.83 | 7.51 | 61.8 | 0.210 | 0.964 |
| **Drift** | 45.06 | 33.93 | 7.53 | 61.8 | 0.007 | 0.968 |
| **Naive** | 46.55 | 34.94 | 7.64 | — | — | 1.000 |
| ARIMA | 47.09 | 35.34 | 7.72 | 52.7 | 0.044 | 1.012 |
| Prophet | 75.55 | 57.39 | 14.75 | 49.2 | 0.162 | 1.623 |

Ensemble weights fitted on validation: LSTM 0.826, Drift 0.092, Prophet 0.068,
XGBoost 0.013, ARIMA 0.000, Naive 0.000.

---

## Reading the directional accuracy

LSTM, XGBoost and the drift baseline all score **61.8343%** — identical to four
decimal places. That is not a coincidence, and it is the most important line
in this report.

Distribution of each model's predicted 30-day return across the test set:

| Model | mean | std | min | max |
|---|---|---|---|---|
| LSTM | +1.98% | **0.03%** | +1.84% | +2.14% |
| XGBoost | +0.97% | **0.13%** | +0.42% | +1.21% |
| Drift | +0.92% | 0.00% | +0.92% | +0.92% |
| ARIMA | −0.06% | 1.71% | −11.47% | +10.25% |
| Prophet | −0.43% | 14.55% | −25.81% | +40.20% |
| *Actual* | *+3.32%* | *9.62%* | | |

The LSTM predicts +1.98% ± 0.03%. After 60-day input windows, two stacked
recurrent layers and early stopping, it has learned to output a constant. GS
rose in 61.83% of the 30-day windows in this test period, so any permanently
bullish forecaster scores 61.83%. All three do.

The drift baseline is two lines of code with no features and no training, and
its RMSE is within 0.4% of XGBoost's. Every technical indicator in
`engineer.py` bought essentially nothing over "assume it drifts up."

The one figure that survives is XGBoost's **IC of 0.210** — rank correlation
between predicted and realised returns. Drift scores 0.007, because a constant
cannot rank anything. So XGBoost has real but modest time-varying signal. The
LSTM's IC of 0.068 alongside its better RMSE says its advantage is almost
entirely a better-tuned drift constant.

`notebooks/plots/08_data_split.png` shows this directly: the distribution of
30-day forward returns in the test window, with the rise rate that every
bullish-constant model reproduces.

---

## Four bugs found and fixed

Each produced results that looked excellent rather than obviously broken, and
each is now covered by a test.

**1. Off-by-one in prediction alignment.** The first run scored XGBoost at
R² = 1.0000, RMSE 0.60, 98.5% directional accuracy. Predictions were indexed by
the *feature* date rather than the *target* date, so a horizon-1 forecast was
graded against a day the model had already seen.
→ `build_feature_frame` returns explicit `target_dates`;
`test_target_dates_are_horizon_ahead` pins it.

**2. Daylight-saving split index.** Source timestamps carried a market-local UTC
offset that flips seasonally (−04:00 summer, −05:00 winter). Summer rows landed
on 04:00 and winter rows on 05:00, so the business-day reindex matched only
half of them and forward-filled the rest — silently dropping 328 of 1,052 test
rows and creating fake flat runs.
→ dates normalised to midnight; `test_loader_normalises_dst_offsets`.

**3. Non-stationary features in tree models.** Trees split on thresholds and
cannot extrapolate: a model trained on prices up to $250 cannot output $914.
With raw levels included, top features were `High` and `SMA_5` and Track B RMSE
was worse than naive.
→ scale-free features only, log-return targets;
`test_xgboost_predicts_log_returns_not_levels`.

**4. Prophet's changepoint range.** Prophet's default `changepoint_range=0.8`
places no changepoints in the most recent 20% of history, so it extrapolated
the trend from 2021 and missed the entire run-up — forecasting 682 against a
last close of 914. Setting `changepoint_range=0.95` moved its serving forecast
to 878 and its test RMSE from 90.55 to 75.55.

Two further serving-path bugs were caught by the API smoke test: the drift
baseline was not compounding with the horizon, and requesting `ensemble` alone
silently collapsed the blend to whichever members happened to be loaded.

---

## Structure

```
gs_stock_prediction/
├── config.py                       # all tunable parameters
├── Makefile                        # make repair / train / serve / test
├── .env.example
├── .gitignore
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── loader.py               # schema-flexible CSV reader
│   │   └── preprocessor.py         # cleaning + leak-safe scaler
│   ├── features/engineer.py        # RSI, MACD, Bollinger, ATR, lags, rolling
│   ├── models/
│   │   ├── baseline.py             # Naive + Drift
│   │   ├── arima_model.py          # AIC grid search, walk-forward append
│   │   ├── prophet_model.py        # rolling-origin refit
│   │   ├── xgboost_model.py        # log-return target
│   │   ├── lstm_model.py           # 2-layer PyTorch, batched rollout
│   │   └── ensemble.py             # SLSQP weights on the simplex
│   └── evaluation/metrics.py       # RMSE/MAE/MAPE + DirAcc, IC, Theil's U
├── pipelines/
│   ├── repair_csv.py               # detects the reversed-column fault
│   ├── fit_ensemble.py             # validation-fitted weights
│   ├── train_pipeline.py           # end-to-end, resumable
│   └── predict_pipeline.py         # inference from saved artefacts
├── api/
│   ├── main.py                     # FastAPI app + lifespan registry
│   ├── schemas.py                  # Pydantic v2
│   └── routers/predict.py          # /forecast /models /metrics /health
├── notebooks/01_EDA_Analysis.py    # 8 diagnostic plots
├── tests/                          # 49 tests
├── deployment/                     # Dockerfile + docker-compose.yml
├── models_saved/                   # trained artefacts
├── data/processed/                 # metrics, forecasts, prediction cache
└── reports/                        # charts + run_manifest.json
```

The loader normalises header names, so exports from Yahoo Finance,
Investing.com, MarketWatch, Nasdaq or yfinance load without editing. It also
handles `1,234.56`, `$412.30`, `12.3M` and `(4.2)` number formats.

---

## API

`uvicorn api.main:app --port 8000`, docs at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Status, loaded models, data recency |
| `GET /api/v1/models` | Per-model availability and test metrics |
| `POST /api/v1/forecast` | N-day forecast, 1–252 days |
| `GET /api/v1/metrics` | Full metrics table from the last training run |

```bash
curl -X POST localhost:8000/api/v1/forecast \
     -H 'Content-Type: application/json' \
     -d '{"steps": 30, "models": ["ensemble"]}'
```

Every `/forecast` response carries the naive and drift baselines alongside the
requested model, plus `model_vs_drift_pct`. That is deliberate: the measured
gap is small enough that a bare number would misrepresent it.

Missing artefacts degrade the service rather than crashing it — `/health`
reports what is live and `/forecast` still serves baselines.

---

## Model notes

**ARIMA(2,1,3)**, selected by AIC (4342.8) over p,q ≤ 3, fitted on the trailing
750 observations. Walk-forward uses `append(refit=False)`. 117s.

**Prophet** fits in log space with multiplicative seasonality and
`changepoint_range=0.95`, refit every 30 business days on a rolling origin.
Still the worst performer (Theil U 1.62) — yearly and weekly seasonality is
close to absent in equity prices, and its ±14.6% forecast spread against a
realised 9.6% shows the trend model overreacting.

**XGBoost** on 60 stationary features. Top features are all oscillators —
`roll_z_20`, `Williams_R`, `Stoch_D`, `ROC_10`, `BB_PctB`. Early stopping fires
after a handful of rounds, which is itself informative.

**LSTM** — 2 layers, hidden 64, 60-day windows over standardised log returns,
Huber loss, early stopping at epoch 8. 35s on CPU. Multi-step evaluation rolls
all origins forward in one batch, turning ~30,000 sequential forward passes
into 30 batched ones.

**Ensemble** — SLSQP on the simplex (w ≥ 0, Σw = 1), fitted on validation.
Best overall RMSE, but note it puts 83% of its weight on the LSTM and 9% on
drift, and lands 3.5% ahead of drift alone.

---

## Caveats

This is a modelling exercise, not investment advice, and nothing here supports
a trading decision. The measured edge over a zero-effort drift baseline is
3.5% RMSE and would not survive transaction costs. Results cover one specific
and strongly bullish test window (2022-2026); a test period spanning a drawdown
would flip the directional accuracy numbers entirely, which is exactly why they
should not be read as skill.

The honest summary of this project is that it is a well-instrumented
demonstration that daily equity prices are close to a random walk — which is
worth more than a leaderboard claiming 98% accuracy.
