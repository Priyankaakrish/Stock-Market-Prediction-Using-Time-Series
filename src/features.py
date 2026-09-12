"""Stage 4: time-series feature engineering.

Two rules govern this module.

1. **Causality.** Every rolling/expanding statistic is computed on data up to
   and including day *t* only. The target is day *t+h*. No centred windows, no
   ``shift(-1)`` anywhere except when building the label itself.

2. **Stationarity.** NVDA's split-adjusted close runs from $0.04 (1999) to
   ~$189 (2026) — a 5,000x range. Gradient-boosted trees cannot extrapolate
   beyond the target range they were trained on, so a model trained on raw
   price levels will predict a ceiling of roughly the highest price it saw in
   training and flat-line for the entire test period. The fix is to learn on
   *scale-free* quantities: log returns, price/MA ratios, %B, normalised
   volume. Those are the columns exposed by :func:`feature_columns`; the raw
   price columns are retained only for reconstruction and plotting.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DATA, LEAKY_COLUMNS, PATHS

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Technical indicators (all causal)
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # An unbroken run of gains gives avg_loss == 0, which makes RS infinite and
    # RSI exactly 100. Dividing straight through yields NaN, and a blanket
    # fillna(50) would silently report a screaming uptrend as neutral.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)

    # Remaining NaNs are the warm-up rows only; 50 is the neutral value and
    # those rows are dropped downstream anyway.
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return upper, mid, lower, width, pct_b


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------
# Feature table
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(PATHS.cleaned)

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    close, high, low, vol = d["Close"], d["High"], d["Low"], d["Volume"]

    # ---- returns and volatility -----------------------------------------
    d["log_ret"] = np.log(close).diff()
    d["ret_1d"] = close.pct_change()
    for k in (5, 10, 21):
        d[f"ret_{k}d"] = close.pct_change(k)
    for w in DATA.vol_windows:
        d[f"volatility_{w}"] = d["log_ret"].rolling(w).std()
    d["vol_ratio"] = d[f"volatility_{DATA.vol_windows[0]}"] / d[
        f"volatility_{DATA.vol_windows[-1]}"
    ].replace(0.0, np.nan)

    # ---- moving averages, kept as ratios to today's close ---------------
    for w in DATA.ma_windows:
        ma = close.rolling(w).mean()
        d[f"MA{w}"] = ma                      # raw, for plots
        d[f"close_over_MA{w}"] = close / ma    # scale-free, for models
    d["MA7_over_MA50"] = d["MA7"] / d["MA50"]
    d["MA20_over_MA50"] = d["MA20"] / d["MA50"]
    d["ema12_over_ema26"] = (
        close.ewm(span=12, adjust=False).mean() / close.ewm(span=26, adjust=False).mean()
    )

    # ---- oscillators -----------------------------------------------------
    d["RSI14"] = rsi(close, DATA.rsi_period)
    macd_line, macd_sig, macd_hist = macd(close, *DATA.macd)
    d["MACD_norm"] = macd_line / close
    d["MACD_signal_norm"] = macd_sig / close
    d["MACD_hist_norm"] = macd_hist / close

    bb_u, bb_m, bb_l, bb_w, bb_p = bollinger(close, DATA.bb_period, DATA.bb_std)
    d["BB_upper"], d["BB_mid"], d["BB_lower"] = bb_u, bb_m, bb_l
    d["BB_width"] = bb_w
    d["BB_pctB"] = bb_p.clip(-0.5, 1.5)

    d["ATR14_norm"] = atr(high, low, close, 14) / close

    # ---- intraday range / candle shape ----------------------------------
    d["hl_range"] = (high - low) / close
    d["oc_change"] = (close - d["Open"]) / d["Open"]
    d["close_pos_in_range"] = (close - low) / (high - low).replace(0.0, np.nan)
    d["gap_open"] = d["Open"] / close.shift(1) - 1.0

    # ---- volume ----------------------------------------------------------
    for w in DATA.ma_windows:
        d[f"Volume_MA{w}"] = vol.rolling(w).mean()
    d["vol_over_MA20"] = vol / d["Volume_MA20"]
    d["log_vol_change"] = np.log(vol.replace(0.0, np.nan)).diff()
    d["dollar_vol_z"] = (
        (vol * close).pipe(np.log)
        .pipe(lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std())
    )

    # ---- lagged returns (lags of the *stationary* series, not the level) --
    for lag in DATA.lags:
        d[f"log_ret_lag_{lag}"] = d["log_ret"].shift(lag)
        d[f"close_over_close_lag_{lag}"] = close / close.shift(lag)

    # ---- calendar --------------------------------------------------------
    # float64, not int. MLflow infers int64 columns as int32 in a model
    # signature and then refuses the int64 input at serving time ("cannot
    # safely convert int64 to int32"), which breaks every call through
    # models:/<name>@Production. One dtype for the whole feature matrix avoids
    # a class of schema-enforcement failures that only appear in deployment.
    d["dow"] = d["Date"].dt.dayofweek.astype("float64")
    d["month"] = d["Date"].dt.month.astype("float64")
    d["is_month_end"] = d["Date"].dt.is_month_end.astype("float64")
    d["is_quarter_end"] = d["Date"].dt.is_quarter_end.astype("float64")    # ---- targets (the ONLY forward-looking columns) ----------------------
    h = DATA.horizon
    d["target"] = close.shift(-h)                       # next-day close level
    d["target_return"] = np.log(close.shift(-h) / close)  # next-day log return

    # Drop the warm-up rows that have NaN indicators and the final h rows that
    # have no label.
    n_before = len(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna().reset_index(drop=True)
    log.info("Feature matrix: %d rows (dropped %d warm-up/edge rows), %d columns",
             len(d), n_before - len(d), d.shape[1])

    if save:
        PATHS.ensure()
        d.to_csv(PATHS.features, index=False)
        log.info("Wrote features -> %s", PATHS.features)
    return d


# Raw-level columns: informative for humans, poison for tree/NN models.
RAW_LEVEL_COLUMNS = {
    "Open", "High", "Low", "Close", "Volume", "Adj Close", "days_since_prev",
    "MA7", "MA20", "MA50", "Volume_MA7", "Volume_MA20", "Volume_MA50",
    "BB_upper", "BB_mid", "BB_lower",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Stationary, scale-free predictors safe to feed to XGBoost / LSTM."""
    return [
        c for c in df.columns
        if c not in LEAKY_COLUMNS and c not in RAW_LEVEL_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    f = build_features()
    print(f"{len(feature_columns(f))} model features")
