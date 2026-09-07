# Goldman Sachs (GS) — Time Series Forecasting, Rebuilt

A walk-forward forecasting study on 26 years of GS daily bars (1999-05-04 → 2026-03-11, 6,755 sessions).

This is a rebuild of an earlier pipeline whose reported metrics were not merely weak but structurally invalid. The headline result changes completely, and so does the conclusion. **The rebuilt models still do not beat a random walk on daily returns — and that is the correct answer, not a failure.**

---

## The problem with the original run

The original `model_comparison.csv` reported this:

| Model | MAE | MAPE | R² | Directional Acc. |
|---|---|---|---|---|
| ARIMA | $186.80 | 47.99% | **−5.15** | 34.0% |
| Prophet | $165.67 | 42.67% | **−3.80** | 30.8% |
| XGBoost | $146.25 | 36.67% | **−3.14** | 56.6% |
| LSTM | — | — | — | 27.4% |

A negative R² means every model was worse than predicting a constant. Directional accuracy below 50% means the sign predictions were worse than a coin flip. Four separate defects produced this:

**1. A 1,013-step forecast was scored as if it were a forecast.**
The pipeline split 70/15/15, fit on the first 70%, then asked ARIMA and Prophet for a single forecast covering the entire test set — four calendar years, in one shot, with no intervening observations. Beyond roughly 10 trading days an ARIMA forecast on equity data converges to its unconditional mean and stays there. Scoring that against a series that nearly tripled is not a measurement of the model.

**2. Tree models were asked to extrapolate a price level.**
XGBoost was trained to predict `Close` directly. Training-era prices ran roughly $50–400; test-era prices ran $300–830. A decision tree predicts by averaging training targets inside a leaf, so it can never output a value above the maximum it was fitted on. Every prediction in the upper half of the test set was clipped at the training ceiling by construction. That single choice explains the $146 MAE.

**3. The LSTM target was scaled off-range.**
`MinMaxScaler` was fit on training prices and applied to test prices. Every test price above the training maximum mapped above 1.0 — outside the range the network ever saw — and the inverse transform then compounded the error. Combined with a sequence-alignment bug that back-filled predictions into the tail of a NaN array, the LSTM row emerged with no metrics at all.

**4. No baseline, so nothing could be interpreted.**
There was no random-walk benchmark anywhere in the pipeline. Without it there is no way to distinguish "this model has an edge" from "this model is elaborate noise". This is the defect that matters most, because it is the one that makes the other three invisible.

---

## What this rebuild changes

| | Original | Rebuild |
|---|---|---|
| Target | `Close` (price level) | next-day **log return** |
| Evaluation | one 1,013-step forecast | **walk-forward, 1-step-ahead**, 1,000 times |
| Refitting | never | every 21 trading days, expanding window |
| Baseline | none | **random walk + drift**, and everything is scored against them |
| Leakage control | implicit | explicit one-day shift + enforced by tests |
| Significance | none | Diebold–Mariano, binomial test on direction |
| Feature scale | raw prices, MAs | ratios, returns, z-scores (scale-free) |
| Verdict metric | price R² | **skill vs random walk**, IC, directional accuracy |

The central mechanical change is one line in `src/features.py`: the entire indicator block is computed on full history, then shifted forward by one row. Features on row `t` therefore contain only what was observable at the close of day `t−1`. `tests/test_pipeline.py` enforces this by recomputing features from a truncated history and asserting the values match.

---

## Results

Walk-forward, 1,000 trading days out-of-sample (2022-03-16 → 2026-03-11). Every model forecasts exactly one day ahead, refit monthly.

### Return space — what the models actually predict

| Model | MAE (bps) | RMSE (bps) | Dir. Acc. | IC | **Skill vs naive** | DM p | Dir p |
|---|---|---|---|---|---|---|---|
| drift | 126.12 | 176.03 | 53.9% | −0.000 | **+0.0017** | 0.158 | 0.007 |
| **naive (random walk)** | 126.37 | 176.18 | — | — | **0.0000** | — | — |
| xgboost | 127.13 | 177.32 | 49.8% | −0.104 | −0.0131 | 0.209 | 0.563 |
| arima | 127.37 | 177.34 | 51.9% | −0.026 | −0.0132 | 0.147 | 0.121 |
| ridge | 127.61 | 177.45 | 50.2% | +0.024 | −0.0145 | 0.154 | 0.462 |
| mlp | 128.11 | 178.03 | 50.2% | −0.055 | −0.0211 | **0.0003** | 0.462 |

**Not one model beats the random walk.** Skill scores are negative for every learned model, information coefficients are indistinguishable from zero, and directional accuracy sits at the coin-flip line. The MLP is significantly *worse* than naive (DM p = 0.0003).

The `drift` row deserves a caveat, because it is the kind of result that gets mistaken for a finding. Drift always predicts a small positive return, so it is always "long". Its 53.9% directional accuracy (p = 0.007) is simply the base rate of up-days in a rising market, not forecasting skill. Its skill score of +0.0017 is economically meaningless.

### Price space — why the original metric was the wrong yardstick

| Model | MAE ($) | MAPE | R² |
|---|---|---|---|
| naive | 5.94 | 1.26% | **0.9977** |
| xgboost | 5.97 | 1.27% | 0.9977 |
| arima | 5.97 | 1.27% | 0.9977 |
| ridge | 6.01 | 1.28% | 0.9977 |

R² went from −5.15 to 0.9977 and MAE from $186.80 to $5.94. That looks like a spectacular fix, and in one sense it is — but the naive baseline scores *identically*. On a one-step horizon, yesterday's close already explains 99.77% of today's variance. A price-space R² near 1.0 is a property of the horizon, not evidence of a model. This is exactly why the baseline had to exist before any number could be read.

### Economically

Long when the forecast is positive, flat otherwise, 1 bp per position change:

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Trades |
|---|---|---|---|---|---|
| **Buy & hold** | **29.2%** | 27.9% | **0.92** | −30.9% | 1 |
| ridge | 20.2% | 21.3% | 0.86 | −23.6% | 312 |
| arima | 18.0% | 23.6% | 0.70 | −37.8% | 433 |
| xgboost | 12.9% | 24.4% | 0.50 | −26.7% | 193 |
| mlp | 10.9% | 19.2% | 0.54 | −23.0% | 144 |

Buy-and-hold wins on both return and Sharpe. Ridge gets closest, with lower drawdown — but that is what being out of the market 45% of the time buys you, and it does not survive as an edge.

---

## Is the pipeline capable of finding signal at all?

A study that finds nothing is only credible if the same machinery finds something when something is there. `pipelines/run_volatility.py` runs the identical walk-forward code on a target that *is* known to be predictable — 22-day forward realised volatility — scored against a matched-horizon persistence benchmark.

| Model | MAE (log) | R² | Corr | Skill vs persistence |
|---|---|---|---|---|
| ridge | 0.236 | **0.209** | 0.516 | **+0.267** |
| xgboost | 0.243 | 0.172 | 0.478 | +0.233 |
| HAR-RV | 0.252 | 0.120 | 0.470 | +0.185 |
| persistence | 0.278 | −0.080 | 0.462 | 0.000 |

Every model beats the benchmark, and the best explains ~21% of the variance in log volatility. The code finds signal where signal exists. Daily returns simply do not contain any.

One caveat found along the way: over the same 2022–2026 window used for the return backtest, volatility persistence collapses — `corr(backward-22d, forward-22d)` falls from 0.70 full-sample to 0.16. Run the control on the longer window (`--test-days 4200`, the default used above) or the control itself looks like a failure for reasons that have nothing to do with the code.

---

## What this means

Daily equity returns on a liquid large-cap name are close to a martingale. That is not a defect in the modelling; it is roughly what an efficient market is supposed to look like, and the result here is consistent with the broad finding in the literature that daily return direction is not reliably predictable from price history alone.

The interesting engineering question is therefore not "which model wins" but "would I have known if none of them did". The original pipeline could not have told you. This one can, and does.

Where the remaining headroom actually is:
- **Volatility and risk** — demonstrably forecastable, and the basis of real products (option pricing, position sizing, VaR).
- **Longer horizons** — monthly and quarterly returns show more documented predictability than daily.
- **Data beyond price** — order flow, earnings revisions, positioning, cross-asset signals. Twenty-five years of OHLCV is a thin diet.
- **Cross-sectional prediction** — ranking many names against each other is a far better-posed problem than forecasting one name's level.

---

## Project structure

```
gs_forecast/
├── config.py                     # all parameters
├── requirements.txt
├── data/
│   ├── raw/gs_master_dataset.csv          # 6,755 daily bars
│   └── processed/
│       ├── walkforward_predictions.csv    # per-day predictions, all models
│       ├── model_comparison.csv           # return + price + significance
│       ├── strategy_comparison.csv        # long/flat backtest
│       └── volatility_*.csv               # positive control
├── src/
│   ├── data.py                   # load, validate OHLC, log returns
│   ├── features.py               # leak-free features (THE one-day shift)
│   ├── models.py                 # naive, drift, ridge, xgboost, mlp, arima
│   ├── models_lstm.py            # optional torch LSTM
│   ├── backtest.py               # walk-forward engine
│   ├── metrics.py                # return/price/significance/strategy
│   └── plots.py
├── pipelines/
│   ├── run_backtest.py           # main study
│   └── run_volatility.py         # positive control
├── tests/test_pipeline.py        # 17 regression tests
└── reports/
    ├── results.txt
    ├── volatility_results.txt
    └── plots/                    # 9 figures
```

## Running it

```bash
pip install -r requirements.txt

python pipelines/run_backtest.py                              # ~1.5 min
python pipelines/run_volatility.py --horizon 22 --test-days 4200
pytest tests/ -v                                              # 17 tests

# knobs
python pipelines/run_backtest.py --test-days 500 --refit-every 5
```

Runtime is ~90 seconds on a single CPU core for the full 1,000-day backtest across six models. The original took 20–40 minutes largely because `auto_arima` searched a wide grid on a 4,000-row non-stationary series; fitting ARIMA on returns with `d=0` over a small AIC grid removes that cost entirely.

## Figures

| | |
|---|---|
| `01_price_and_split.png` | price history with the out-of-sample boundary |
| `02_stationarity.png` | ACF of price vs returns, return distribution, rolling vol |
| `03_price_paths.png` | reconstructed one-step price paths (all models overlap) |
| `04_return_scatter.png` | predicted vs realised return — the honest view |
| `05_equity_curves.png` | long/flat strategies vs buy-and-hold |
| `06_rolling_dir_acc.png` | rolling 126-day directional accuracy vs 50% |
| `07_feature_importance.png` | XGBoost gain, final refit |
| `08_old_vs_new.png` | original R² vs rebuild, with the naive bar included |
| `09_volatility.png` | HAR forecast vs realised volatility |

---

*Research code. Not investment advice, and nothing here constitutes a trading recommendation — the most defensible finding in it is that this class of model does not predict daily GS returns.*
