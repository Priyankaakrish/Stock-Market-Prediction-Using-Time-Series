# Goldman Sachs Stock Price Prediction — Time Series

ARIMA, Prophet, XGBoost, LSTM and a weighted ensemble, benchmarked against a
random walk, with a FastAPI service and Docker deployment. 
Data: `goldmansachs.csv`, 6,709 rows, 1999-05-04 to 2026-01-02, after
repairing a column-ordering fault in the source file.

---

## Headline result

**No model beats a random walk with drift by a meaningful margin.** The best
model (XGBoost) improves RMSE by 0.4% over a two-line baseline with zero parameters.
Reported directional accuracy of 61.83% is the test window's upward drift, not
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

Run environment: Windows 11, Python 3.14.0, pandas 3.0.5, numpy 2.5.2,
statsmodels 0.15.0, xgboost 3.4.1, prophet 1.4.0.

Split: train 1999-05-04 to 2018-01-02 (4,871 days), validation 2018-2022
(1,044), test 2022-01-04 to 2026-01-02 (1,044). Chronological, never shuffled.

### Track B - 30 business days ahead (n = 1,014)

| Model | RMSE | MAE | MAPE % | R2 | DirAcc % | IC | Theil U |
|---|---|---|---|---|---|---|---|
| XGBoost | 44.87 | 33.85 | 7.51 | 0.928 | 61.83 | **0.248** | 0.964 |
| **Drift** | 45.06 | 33.93 | 7.53 | 0.928 | 61.83 | 0.007 | 0.968 |
| **Naive** | 46.55 | 34.94 | 7.64 | 0.923 | - | - | 1.000 |
| ARIMA(2,1,3) | 47.09 | 35.34 | 7.72 | 0.921 | 52.66 | 0.044 | 1.012 |
| Prophet | 75.46 | 57.40 | 14.76 | 0.797 | 49.11 | 0.158 | 1.621 |

### Track A - 1 day ahead (n = 1,044)

| Model | RMSE | MAE | MAPE % | R2 | DirAcc % | IC | Theil U |
|---|---|---|---|---|---|---|---|
| XGBoost | 7.990 | 5.278 | 1.206 | 0.998 | 52.99 | 0.013 | 0.999 |
| **Drift** | 7.994 | 5.283 | 1.207 | 0.998 | 53.04 | -0.033 | 0.999 |
| **Naive** | 8.004 | 5.287 | 1.207 | 0.998 | - | - | 1.000 |
| ARIMA(2,1,3) | 8.062 | 5.390 | 1.234 | 0.998 | 47.46 | -0.015 | 1.007 |

All four models sit within 1% of each other. ARIMA is worse than doing nothing
(Theil U above 1.0).

The LSTM and ensemble rows are absent for environment reasons - see
*Limitations* below.

---

## Reading the directional accuracy

XGBoost and the drift baseline both score **61.8343%** - identical to four
decimal places. That is not a coincidence and it is the central finding.

Drift is `last_close x exp(mu x horizon)` with `mu` estimated on training data:
one parameter, no features, no training loop. It predicts +0.92% over every
30-day window in the test period. GS rose in 61.83% of those windows, so any
permanently bullish forecaster scores exactly 61.83%. XGBoost's near-constant
prediction reproduces the same figure.

Supporting diagnostics from `notebooks/01_EDA_Analysis.py`:

- ADF on price level: p = 1.000 (non-stationary)
- ADF on log returns: p = 3.6e-26 (stationary)
- Lag-1 return autocorrelation: **-0.0467** - almost no short-horizon structure
- Test-window 30-day rise rate: **61.83%**

The figure that survives is XGBoost's **IC of 0.248** - Spearman rank
correlation between predicted and realised returns - against drift's 0.007. A
constant cannot rank anything, so this is genuine, if modest, time-varying
signal. It is the only metric in either table where a trained model clearly
separates from a zero-effort baseline.

XGBoost's early stopping fired at `best_iteration=0`: validation loss rose on
the very first boosting round. That is consistent with the above rather than a
contradiction of it - there is little learnable signal, so the fitted model is
close to a constant.

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
├── .env.example
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
750 observations. Walk-forward uses `append(refit=False)`. 278s in this run.

**Prophet** fits in log space with multiplicative seasonality and
`changepoint_range=0.95`, refit every 30 business days on a rolling origin.
Still the worst performer (Theil U 1.62, 400s in this run) — yearly and weekly seasonality is
close to absent in equity prices, and its ±14.6% forecast spread against a
realised 9.6% shows the trend model overreacting.

**XGBoost** on 60 stationary features. Top features in this run:
`RSI`, `roll_z_50`, `Price_SMA_200_ratio`, `Volume_ratio`, `DayOfWeek`. Early
stopping fired at `best_iteration=0` — validation loss rose on the first
boosting round, which is itself informative.

**LSTM** — 2 layers, hidden 64, 60-day windows over standardised log returns,
Huber loss, early stopping. Multi-step evaluation rolls all origins forward in
one batch, turning ~30,000 sequential forward passes into 30 batched ones. Not
evaluated in this run — see *Limitations*.

**Ensemble** — SLSQP on the simplex (w ≥ 0, Σw = 1), fitted on validation.
Weights came out LSTM 0.826, Drift 0.092, Prophet 0.068, XGBoost 0.013, with
ARIMA and Naive at zero. Not evaluated in this run — see *Limitations*.

---

## Limitations

**LSTM and ensemble not reproduced.** PyTorch's Windows wheel installs without
`fbgemm.dll` and `asmjit.dll`, which `torch_cpu.dll` requires, so `import torch`
fails with WinError 126. Confirmed as an environment issue rather than a code
one: clean `--force-reinstall --no-cache-dir` under both Python 3.14 and 3.12,
with an intact VC++ runtime (`vcruntime140.dll`, `vcruntime140_1.dll`,
`msvcp140.dll` all present) and no admin rights to set an antivirus exclusion.
`torch_cpu.dll` extracts at 305 MB; the two dependencies never appear.

The code paths are unchanged and run elsewhere. `EnsembleForecaster.predict()`
refuses to blend when members holding more than 50% of the fitted weight are
absent, so the ensemble is omitted rather than reported as a different model
under the same name.

## Caveats

This is a modelling exercise, not investment advice, and nothing here supports
a trading decision. The measured edge over a zero-effort drift baseline is
0.4% RMSE and would not survive transaction costs. Results cover one specific
and strongly bullish test window (2022-2026); a test period spanning a drawdown
would flip the directional accuracy numbers entirely, which is exactly why they
should not be read as skill.

The honest summary of this project is that it is a well-instrumented
demonstration that daily equity prices are close to a random walk — which is
worth more than a leaderboard claiming 98% accuracy.
