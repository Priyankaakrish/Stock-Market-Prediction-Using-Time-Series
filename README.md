# Goldman Sachs Stock Price Prediction — Time Series

ARIMA, Prophet, XGBoost, LSTM and a weighted ensemble, benchmarked against a
random walk, with a FastAPI service and Docker deployment.
with the evaluation methodology tightened so the numbers mean something.

Data: `goldmansachs.csv`, 6,709 rows, 1999-05-04 to 2026-01-02, after
repairing a column-ordering fault in the source file.

# Results

Run on `goldmansachs.csv` (6,709 rows, 1999-05-04 to 2026-01-02) after repairing
the reversed-column fault in the source file. Windows 11, Python 3.14,
pandas 3.0.5, numpy 2.5.2, statsmodels 0.15.0, xgboost 3.4.1, prophet 1.4.0.

Split: train 1999-05-04 → 2018-01-02 (4,871 days), validation 2018-2022 (1,044),
test 2022-01-04 → 2026-01-02 (1,044). Chronological, never shuffled.

---

## Headline

**No model beats a random walk with drift by a meaningful margin.** XGBoost, the
best performer, improves 30-day RMSE by 0.4% over a two-line baseline with zero
parameters and no features. Reported directional accuracy of 61.8% is the test
window's upward drift, not forecasting skill.

---

## Track B — 30 business days ahead (n = 1,014)

| Model | RMSE | MAE | MAPE % | R² | DirAcc % | IC | Theil U |
|---|---|---|---|---|---|---|---|
| XGBoost | 44.87 | 33.85 | 7.51 | 0.928 | 61.83 | **0.248** | 0.964 |
| **Drift** | 45.06 | 33.93 | 7.53 | 0.928 | 61.83 | 0.007 | 0.968 |
| **Naive** | 46.55 | 34.94 | 7.64 | 0.923 | — | — | 1.000 |
| ARIMA(2,1,3) | 47.09 | 35.34 | 7.72 | 0.921 | 52.66 | 0.044 | 1.012 |
| Prophet | 75.46 | 57.40 | 14.76 | 0.797 | 49.11 | 0.158 | 1.621 |

## Track A — 1 day ahead (n = 1,044)

| Model | RMSE | MAE | MAPE % | R² | DirAcc % | IC | Theil U |
|---|---|---|---|---|---|---|---|
| XGBoost | 7.990 | 5.278 | 1.206 | 0.998 | 52.99 | 0.013 | 0.999 |
| **Drift** | 7.994 | 5.283 | 1.207 | 0.998 | 53.04 | −0.033 | 0.999 |
| **Naive** | 8.004 | 5.287 | 1.207 | 0.998 | — | — | 1.000 |
| ARIMA(2,1,3) | 8.062 | 5.390 | 1.234 | 0.998 | 47.46 | −0.015 | 1.007 |

All four models sit within 1% of each other. ARIMA is worse than doing nothing
(Theil U > 1).

The LSTM is absent from both tables: PyTorch's cp314 Windows wheels fail to load
(`shm.dll`, WinError 126). The ensemble is absent because 83% of its fitted
weight sits on the LSTM, and a blend missing most of its weight is a different
model wearing the same name — `EnsembleForecaster.predict()` refuses rather than
reporting it.

---

## How to read the directional accuracy

XGBoost and the drift baseline both score **61.8343%** — identical to four
decimal places. That is not a coincidence and it is the central finding.

Drift is `last_close × exp(mu × horizon)` with `mu` estimated on training data:
one parameter, no features, no training loop. It predicts +0.92% over every
30-day window in the test period. GS rose in 61.83% of those windows, so any
permanently bullish forecaster scores exactly 61.83%. XGBoost's near-constant
+0.97% prediction reproduces the same figure.

Supporting diagnostics from `notebooks/01_EDA_Analysis.py`:

- ADF on price level: p = 1.000 (non-stationary)
- ADF on log returns: p = 3.6e-26 (stationary)
- Lag-1 return autocorrelation: **−0.0467** — almost no short-horizon structure
- Test-window 30-day rise rate: **61.83%**

The one figure that survives is XGBoost's **IC of 0.248** — Spearman rank
correlation between predicted and realised returns — against drift's 0.007. A
constant cannot rank anything, so this is genuine, if modest, time-varying
signal. It is also the only metric in either table where a trained model is
clearly separated from a zero-effort baseline.

XGBoost's early stopping fired at `best_iteration=0`, meaning validation loss
rose on the very first boosting round. That is consistent with the above rather
than a contradiction of it: there is little learnable signal, so the fitted model
is close to a constant.

---

## Reproducibility note

These numbers were produced independently of the reference run, on different
hardware and different library versions (pandas 3.0.5 vs 3.0.2, numpy 2.5.2 vs
2.4.4). The reference run reported XGBoost at 44.89 RMSE / IC 0.210; this run
gives 44.87 / 0.248. ARIMA's AIC grid search independently selected the same
(2,1,3) order. The 61.8343% tie between XGBoost and drift reproduced exactly,
which is expected — it is a property of the test window, not of any particular
fit.

---

## Caveat

This is a modelling exercise, not investment advice. The measured edge over a
zero-effort drift baseline is 0.4% RMSE and would not survive transaction costs.
Results cover one specific and strongly bullish test window (2022-2026); a test
period spanning a drawdown would invert the directional accuracy figures, which
is precisely why they should not be read as skill.
