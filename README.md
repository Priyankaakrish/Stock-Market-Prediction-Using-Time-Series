# NVDA Stock Price Forecasting — End-to-End Time-Series ML Pipeline

Production-shaped pipeline for one-day-ahead NVIDIA close-price forecasting:
ingestion → cleaning → feature engineering → chronological splitting → seven
models → MLflow tracking and registry → FastAPI serving → Docker/nginx →
CI/CD to EC2.

## The headline result, stated up front

**No model in this project beats a random walk to any statistically
significant degree.** On a 1,019-day held-out window, the best model improves
RMSE over "tomorrow's price equals today's price" by 0.08%, with a
Diebold-Mariano p-value of 0.62. The apparent 53.8% directional accuracy is
fully explained by the base rate of up-days in the test window (53.78%) — the
winning model achieves it by always predicting *up*.

This is the correct finding for daily equity closes, not a failure of the
pipeline. Published stock-prediction results claiming sub-1% MAPE are usually
reporting a lagged copy of the input, which is exactly what the `naive`
baseline row measures. The value of this repository is the harness that makes
that visible: the baselines, the significance test, and the base-rate
comparison.

## Results

Test window: 2022-03-18 → 2026-04-10 (1,019 trading days). All forecasts are
strict one-step-ahead.

| Model     | RMSE ($) | MAE ($) | MAPE (%) | Directional acc. | Skill vs naive | DM p-value |
|-----------|---------:|--------:|---------:|-----------------:|---------------:|-----------:|
| drift     | **2.998** | **1.914** | **2.382** | 53.83% | +0.08% | 0.62 |
| lstm      | 2.998 | 1.914 | 2.384 | 53.54% | +0.06% | 0.74 |
| naive     | 3.000 | 1.919 | 2.386 | undefined | — | — |
| xgboost   | 3.001 | 1.916 | 2.383 | 51.87% | −0.04% | 0.91 |
| arima     | 4.084 | 2.705 | 3.423 | 49.02% | −36.1% | <0.001 |
| ma5       | 4.230 | 2.856 | 3.619 | 49.41% | −41.0% | <0.001 |
| prophet   | 26.23 | 19.81 | 27.21 | 50.10% | −774% | <0.001 |

Up-day base rate in the test window: **53.78%**.

Reading the table:

- **drift, lstm, naive, xgboost are one cluster.** Their RMSEs differ in the
  third decimal place. DM p-values above 0.6 say the differences are noise.
- **Naive has no directional accuracy.** It predicts zero change every day, so
  it never commits to a direction. Reporting 0% would be misleading; the
  harness returns `NaN`.
- **ARIMA and MA5 are genuinely worse** than the random walk, significantly so.
  ARIMA(2,1,2)'s ARMA terms fit noise in the log-return series and add variance.
- **Prophet fails badly, and informatively.** It imposes a smooth trend plus
  weekly/yearly seasonality on a series that is close to a martingale. Between
  quarterly refits its trend extrapolation drifts far from the realised price.
  A 27% MAPE is what structural mismatch looks like.

## Two decisions that determined the outcome

### 1. Predict returns, not price levels

The chronological split puts the training window at a maximum close of **$6.23**
while the test window reaches **$207.04** — a consequence of NVDA's ~5,000x
split-adjusted appreciation since 1999.

Gradient-boosted trees predict the mean of a leaf, so their output is bounded
by the target range seen in training. An XGBoost trained on price *levels*
would emit a flat line near $6 for the entire test period, scoring an RMSE
around $70 while appearing to have trained successfully. A min-max-scaled LSTM
has the mirror-image problem: every test price maps above 1.0, into a region
the network never saw.

Both models therefore learn `log(Close_t+1 / Close_t)` and the price forecast
is reconstructed as `Close_t × exp(ŷ)`. `tests/test_pipeline.py::TestXGBoost::
test_predicts_returns_not_levels` is a regression guard against anyone undoing
this.

### 2. The feature matrix excludes raw price levels

`feature_columns()` filters out `Close`, `MA7`, `BB_upper`, `Volume_MA20` and
every other non-stationary level, exposing only scale-free quantities:
`close_over_MA20`, `BB_pctB`, `RSI14`, `MACD_norm`, lagged log returns,
`vol_over_MA20`. 46 features survive. Levels are retained in the dataframe for
plotting and reconstruction, but never reach a model.

## Architecture

```
data/raw/nvda_raw.csv ──▶ preprocess ──▶ features (46) ──▶ chronological split
                                                                │
                        ┌───────────────────────────────────────┘
                        ▼
   baselines (naive / ma5 / drift) · ARIMA · Prophet · XGBoost · LSTM
                        │
                        ▼
   evaluate (RMSE/MAE/MAPE/directional/Diebold-Mariano)
                        │
                        ▼
   MLflow (SQLite backend) ──▶ Model Registry: nvda-forecaster
                        │
                        ▼
   models/best_model.pkl ──▶ FastAPI ──▶ nginx ──▶ EC2
```

## Quick start

```bash
pip install -r requirements-dev.txt

make data          # ingest -> clean -> features -> split
make train         # all 7 models, ~3 min
make test          # 39 tests
make api           # http://localhost:8000/docs
make mlflow        # http://localhost:5002
```

Docker stack (API + MLflow + nginx on the `nvda-net` bridge):

```bash
docker compose -f deploy/docker-compose.yml up -d --build
curl localhost/health
curl -X POST localhost/forecast -H 'Content-Type: application/json' -d '{}'
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness, model and data status. Used by the container healthcheck and the ALB. |
| `/model` | GET | Deployed model metadata and its held-out metrics. |
| `/forecast` | POST | Next-day close. Empty body uses the server's cached history; supply `bars` (≥120) to forecast from your own data. |
| `/metrics` | GET | Prometheus text format. Restricted to private ranges in the nginx config. |
| `/docs` | GET | OpenAPI UI. |

```json
{
  "ticker": "NVDA",
  "model": "drift",
  "as_of": "2026-04-13",
  "last_close": 189.31,
  "predicted_close": 189.5097,
  "predicted_change_pct": 0.1055,
  "direction": "up",
  "prediction_interval_80": [183.7502, 195.2692],
  "disclaimer": "Model output for research and educational use only..."
}
```

The 80% interval is computed from realised 60-day volatility, **not** from the
model. None of these point forecasters produces a calibrated predictive
distribution, and presenting one as if it did would be dishonest.

## Repository layout

```
src/
  config.py              paths and hyperparameters, single source of truth
  ingest.py              yfinance download, retries, validation, CSV fallback
  preprocess.py          dedupe, sort, forward-fill, zero-volume repair
  features.py            46 causal, scale-free features
  split.py               chronological split + walk-forward windows
  evaluate.py            metrics, Diebold-Mariano, comparison table
  train.py               orchestrator: MLflow tracking, selection, registry
  plots.py               7 evaluation figures
  mlflow_model.py        pyfunc wrapper for registry deployment
  models/                base · baselines · arima · prophet · xgboost · lstm
  spark_jobs/            PySpark/EMR feature engineering with parity check
api/                     FastAPI service and pydantic schemas
deploy/                  Dockerfiles, compose, nginx, AWS bootstrap/IAM
tests/                   39 tests, leakage checks first
reports/                 metrics.csv, predictions, figures, EVALUATION.md
```

## Leakage controls

Look-ahead bias in a price pipeline does not raise an exception — it produces a
99.9% R² and a model that loses money. The controls:

1. **Chronological splits only.** `chronological_split` asserts
   `train.max < val.min < test.max`. A random split is the single most common
   error in this problem class.
2. **Causal indicators.** Every rolling window is trailing. A test perturbs the
   final close and asserts no earlier indicator value changes.
3. **Target isolation.** `target` and `target_return` are the only forward-
   looking columns and are excluded from the feature matrix by name.
4. **Correlation tripwire.** A test fails if any feature correlates with the
   target above |0.99|.
5. **Honest backtests.** ARIMA appends realised values via `append(refit=False)`
   so each forecast is genuinely one-step-ahead with fixed coefficients.
   Prophet retrains on a quarterly cadence and rolls forward — never fitted on
   the window it is scored against.
6. **Scaler fitted on train only.** The LSTM's standardisation statistics come
   from the training fold; validation windows are built by prepending training
   rows for context.

## Deployment

- **CI/CD** (`.github/workflows/ci-cd.yml`): lint → 39 tests → smoke-train →
  model quality gate → build → Trivy scan → ECR push → SSM deploy → health check.
- **Quality gate** blocks the deploy if the champion's RMSE exceeds 1.05× naive
  or MAPE exceeds 5%. It is a regression gate, not a "must beat the market" gate.
- **No SSH.** Deployments go over AWS SSM; port 22 stays closed. GitHub
  authenticates via OIDC, so there are no long-lived AWS keys in CI.
- **Serving image excludes torch and prophet.** Shipping a 2 GB CUDA stack to
  serve a model that is a few hundred bytes of coefficients is waste. If an
  LSTM is ever promoted, uncomment the CPU-only torch line in
  `requirements-serve.txt`.

## Known limitations

- **Horizon is one day.** Multi-step forecasting would need recursive
  prediction with error accumulation, or direct multi-horizon models.
- **Prophet's cadence is a choice, not a tuning result.** Quarterly refits are
  realistic for production; daily refits would score better and cost ~1,000
  Stan fits per evaluation.
- **ARIMA order is fixed at (2,1,2).** `grid_search_order` exists and selects
  by AIC on the training fold, but AIC on log-returns is nearly flat across
  small orders, so this buys little.
- **No transaction costs or slippage.** Directional accuracy is not a trading
  strategy. A 53.8% hit rate at retail spreads is not profitable.
- **Single asset, single regime.** The test window spans one exceptional bull
  run. These numbers would not transfer to a different asset or period.
- **The Spark path is written but not executed here** — no Spark runtime was
  available in the build environment. Its parity check (`--validate-against`)
  is the mechanism for verifying it against the pandas implementation.

## Disclaimer

Research and educational use only. Nothing here is investment advice. A
next-day price forecast — particularly one that does not beat a random walk —
must not be used to make trading decisions.
