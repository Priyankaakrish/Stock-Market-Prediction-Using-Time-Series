# Evaluation Report — NVDA One-Day-Ahead Close Forecasting

Generated from `python -m src.pipeline --mode local --walk-forward --promote`.
Figures in `reports/figures/`, raw numbers in `reports/metrics.csv` and
`reports/walkforward_folds.csv`.

---

## 1. Data

| | |
|---|---|
| Source | `nvidia_stock.csv` (Yahoo Finance OHLCV, split-adjusted) |
| Rows | 6,847 daily bars |
| Span | 1999-01-22 → 2026-04-13 |
| Missing values | 0 |
| Duplicate dates | 0 |
| Close range | $0.0366 → $207.04 |

Quality was high enough that cleaning was nearly a no-op: no forward-fills, no
duplicates, no zero-volume repairs fired. Those paths remain because a live
`yfinance` pull is not this clean.

**No calendar reindex.** Stock prices do not exist on weekends and holidays.
Forward-filling them would invent ~30% fake observations, flatten realised
volatility and corrupt every rolling statistic downstream. The trading-day
sequence is the correct index; the pipeline reports weekday share on every run
(`100.0% weekdays`) so a regression here is visible immediately.

## 2. Features

46 model-facing columns from 6,787 usable rows, out of 65 total. The gap
matters: raw price levels stay in the table for plotting and reconstruction but
never reach a model.

| Group | Count | Examples |
|---|---:|---|
| MA ratios | 6 | `close_over_MA7/20/50`, `MA7_over_MA50` |
| Oscillators | 6 | `RSI14`, `MACD_norm`, `BB_pctB` |
| Lags | 14 | `log_ret_lag_{1,2,3,5,7,14,30}` |
| Volatility | 3 | `volatility_7/21`, `ATR14_norm` |
| Returns, volume, candle, calendar | 17 | `ret_5d`, `vol_over_MA20`, `gap_open`, `dow` |

Two rules govern the module. **Causality**: every rolling statistic uses data up
to and including day *t*; a test perturbs the final close and asserts no earlier
indicator value changes. **Stationarity**: `feature_columns()` exposes only
scale-free quantities, for the reason in §3.

## 3. Splits — and the fact that determined everything

| Split | Rows | Start | End | Close min | Close max |
|---|---:|---|---|---:|---:|
| train | 4,750 | 1999-04-19 | 2018-03-02 | $0.03 | $6.23 |
| val | 1,018 | 2018-03-05 | 2022-03-17 | $3.18 | $33.38 |
| test | 1,019 | 2022-03-18 | 2026-04-10 | $11.23 | $207.04 |

**Training tops out at $6.23; the test window reaches $207.04 — a 33x level
gap.** This is the single most consequential number in the project.

Gradient-boosted trees predict the mean of a leaf, so output is bounded by the
training target range. An XGBoost trained on price *levels* would emit a flat
line near $6 for the entire test period — RMSE around $70 — while appearing to
have trained successfully. A min-max-scaled LSTM has the mirror problem: every
test price maps above 1.0, into a region the network never saw.

Both therefore learn `log(Close_t+1 / Close_t)` and reconstruct price as
`Close_t * exp(pred)`. `TestXGBoost::test_predicts_returns_not_levels` guards it.

## 4. Metrics

RMSE and MAE in dollars, dominated by the recent high-price regime. MAPE is
scale-free and tells the same story. Directional accuracy returns `NaN` for
models that never commit to a direction. Diebold-Mariano on squared-error loss
with Newey-West variance and the Harvey small-sample correction.

## 5. Single-split results

Test window 2022-03-18 to 2026-04-10, 1,019 days, strict one-step-ahead.

| Model | RMSE | MAE | MAPE | Dir. acc. | Skill vs naive | DM p |
|---|---:|---:|---:|---:|---:|---:|
| drift | **2.998** | **1.914** | **2.382** | 53.83% | +0.077% | 0.622 |
| lstm | 2.998 | 1.914 | 2.384 | 53.54% | +0.055% | 0.737 |
| xgboost | 2.999 | 1.917 | 2.382 | 52.16% | +0.027% | 0.938 |
| naive | 3.000 | 1.919 | 2.386 | undefined | — | — |
| arima | 4.083 | 2.704 | 3.423 | 49.02% | −36.1% | <0.001 |
| ma5 | 4.230 | 2.856 | 3.619 | 49.41% | −41.0% | <0.001 |
| prophet | 26.20 | 19.78 | 27.22 | 50.10% | −773% | <0.001 |

**Up-day base rate: 53.78%.**

drift, lstm, xgboost and naive differ in the third decimal. DM p-values above
0.6 say the differences are noise. Drift's 53.83% directional accuracy is the
up-day base rate to two decimal places — it predicts up every single day.

ARIMA and MA5 are *significantly worse*. Prophet fails by a factor of nine: it
imposes trend-plus-seasonality on a near-martingale, and between quarterly
refits its extrapolation drifts far from the realised path. That is what
structural model mismatch looks like.

## 6. Walk-forward validation

A single split gives one number per model and cannot separate a real edge from
a lucky window. 16 expanding-window folds of ~63 trading days, each model
refitted from scratch.

| Model | RMSE mean | RMSE std | Folds beaten naive | Binomial p |
|---|---:|---:|---:|---:|
| drift | 2.513 | 1.675 | 10 / 16 | 0.454 |
| xgboost | 2.514 | 1.672 | 10 / 16 | 0.454 |
| naive | 2.515 | 1.673 | — | — |
| arima | 3.452 | 2.200 | 0 / 16 | <0.001 |
| ma5 | 3.575 | 2.280 | 0 / 16 | <0.001 |

**The strongest evidence in the project.** Ten wins out of sixteen; a coin flip
gives eight, p = 0.454. Directional accuracy swings ±5.8 pp fold to fold —
twenty times the 0.05 pp by which drift exceeds the base rate.

ARIMA and MA5 lose in **every fold**. Sixteen consecutive losses is structure,
not luck.

Two details only visible because folds refit independently:

**XGBoost's optimal tree count across folds:** 92, 208, 58, 198, 236, 181, 61,
15, 44, 44, 41, 43, 43, 42, 39, 14. Early stopping picks anywhere from 14 to 236
trees on nearly identical data. A model whose optimal complexity swings
seventeen-fold between adjacent quarters is fitting noise and halting wherever
the validation fold happens to bottom out.

**Worst-fold RMSE is roughly double the mean** for every model. That is the
price level rising, not degradation — dollar RMSE is not comparable across
folds. MAPE is, and stays near 2.4%.

`_fresh()` rebuilds each model per fold. An ARIMA that has already absorbed
fold 3's observations is not a fair fold-4 model.

## 7. Drift detection

PSI with quantile bins plus a KS test, training vs test distribution.
**4 of 46 features significant, 3 moderate.**

| Feature | PSI | Mean shift (sigma) | Severity |
|---|---:|---:|---|
| volatility_21 | 1.483 | −0.139 | significant |
| ATR14_norm | 0.808 | −0.222 | significant |
| log_vol_change | 0.271 | −0.004 | significant |
| vol_over_MA20 | 0.262 | −0.041 | significant |
| volatility_7 | 0.185 | −0.098 | moderate |
| BB_width | 0.171 | −0.137 | moderate |
| hl_range | 0.160 | −0.210 | moderate |

**Every significantly-drifted feature is a volatility or range measure, and
every shift is negative.** Training spans the dot-com collapse and 2008; the
test window does not. The models were fitted on a substantially wilder version
of this stock than the one they forecast.

Momentum and moving-average ratios all sit below the 0.10 threshold. Trend
structure is unchanged; only amplitude moved.

KS p-values are ~0 even for stable features. At n = 4,750 the test detects
shifts too small to matter, which is why PSI carries the severity call.

**This closes the loop.** Seven models found no exploitable signal in returns.
Drift analysis then identifies volatility as the one quantity that measurably
changed — and volatility is the one quantity the literature says is
predictable, because it clusters. §10's first recommendation is not a guess.

## 8. Serving-side monitoring

`/predictions` summarises recent forecasts:

```json
{"n": 1, "mean_change_pct": 0.10548, "share_up_pct": 100.0,
 "implausible_rate_pct": 0.0}
```

`share_up_pct: 100.0` is the same finding from a third angle: the deployed
model is bullish on every call. Statistical (§6), structural (§7), operational
(§8) — three independent views converge.

`implausible_rate_pct` is the alarm signal rather than the mean. Of seven
CloudWatch alarms, two are specific to this project: one fires when >1% of
forecasts imply a move beyond ±10%, the other when live RMSE exceeds the random
walk on the same window — because a model losing to naive has negative value
and should be rolled back.

## 9. Five bugs that only appeared when the code ran

Each passed review and sat in the repository looking correct.

**Dead walk-forward.** `walk_forward_windows()` existed in `split.py`, was never
called, and no test touched it. The diagram requires walk-forward validation, so
the repository had a function whose only purpose was to make that box look
ticked.

**Spark windows without min-periods.** First execution produced 6,846 rows
against pandas' 6,787. pandas' `rolling(50)` returns NaN until 50 observations
exist; Spark's `rowsBetween(-49, 0)` silently averages what is available, so
row 3 receives a "50-day average" from three observations. No error — just 59
rows of wrong indicators at the head of the series. After a `WARMUP_ROWS` guard:
6,787 both sides, agreement to 1e-16. The parity check itself had two bugs — a
dtype mismatch that raised before comparing, and value-only comparison that
passed while an inner join silently dropped 59 rows.

**The registry serving path never worked.** Every run printed
`UserWarning: Inferred schema contains integer column(s)`, dismissed as noise.
The calendar features were logged as int32, so schema enforcement rejected every
inference call: `Can not safely convert int64 to int32`.
`models:/nvda-forecaster@Production` — documented in `.env.example` and offered
as the production serving mode — **failed on every call from the day it was
written**. Nothing caught it because nothing ever resolved the alias, and the
API's fallback to a local pickle kept the failure invisible. Fixed at source:
the feature matrix now carries one dtype throughout.

**The pipeline did everything twice.** After splitting the orchestrator into
per-stage timings, `train.run()` still called `build_dataset()` and rebuilt
ingest, preprocess, features and split from scratch. Visible in the logs as a
second `Feature matrix: 6787 rows` seven seconds after the first. Fixed with
`reuse=True`, tested by asserting `ingest` is never called.

**The API could not serve from the registry.** `_load_model()` recorded *what*
it loaded but not *which kind*. The registry returns a `PyFuncModel` exposing
only `predict`; the pickle returns a `Forecaster` exposing `backtest`. The
forecast handler called `backtest` unconditionally, so setting
`MODEL_URI=models:/nvda-forecaster@Production` — the production serving mode
documented in `.env.example` — returned HTTP 500 on every request. The registry
branch also returned before loading `best_model_meta.json`, so `/health`
reported a null model name. Every existing test ran in pickle mode, so nothing
saw it. Fixed by recording the model kind and dispatching; four tests now run
the API in registry mode, including one asserting both modes produce the same
number.

The pattern is consistent: plausible code that has never been exercised. What
found the schema bug was building the promotion gate, because promoting forced
something to resolve the alias for the first time. The `verify serving path`
stage now makes one real inference on every promotion — through the registry
*and* through the API, since proving the registry can serve is not the same as
proving the service can. It is tested against a broken signature, a 500 from the
API, and an implausible forecast.

A note on that stage: its first version asserted the API and registry prices
must match exactly, and failed immediately with `189.5097 vs 188.8290`. Not a
bug — the registry call forecasts from the feature table, whose last row is the
last day that *has a label*, while the API pads its cached bars so the true
final bar survives. They legitimately forecast different days. The check now
verifies plausibility rather than equality. A check that fires on correct
behaviour is worse than no check, because it trains you to ignore it.

## 10. What would actually be worth trying

1. **Change the target.** Volatility clusters and is forecastable at daily
   horizons; returns are not. A GARCH-family model on this same data would
   likely be the first model here to show real skill. §7 points directly at it.
2. **Change the horizon.** Weekly or monthly aggregation reduces microstructure
   noise, and momentum has empirical support there.
3. **Change the features.** Price and volume are the most-mined data in
   existence. Options-implied volatility, earnings surprises or supply-chain
   data carry information not already in the price.
4. **Change the question.** "Will tomorrow's move exceed ±2%" is more tractable
   than predicting the price.
5. **Reconsider the split.** A 1999-2018 training window teaches the model about
   a company that no longer exists in the same form.

What is *not* worth trying: more technical indicators, deeper networks, more
hyperparameter search. Flat feature importances, DM p-values above 0.6, and a
tree count that swings seventeen-fold between quarters all say the same thing.
