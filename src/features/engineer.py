"""
src/features/engineer.py
─────────────────────────
Industry-standard technical-indicator feature engineering for
NVIDIA stock price prediction.

Indicators implemented (pure-pandas, no TA-Lib dependency):
  • Trend   : SMA, EMA, MACD, MACD Signal, MACD Histogram
  • Momentum: RSI, Stochastic %K/%D, Williams %R, ROC
  • Volatility: Bollinger Bands (upper/mid/lower/width/%B), ATR, Historical Vol
  • Volume   : OBV, VWAP, Volume SMA ratio
  • Calendar : day-of-week, month, quarter, is-month-end, trading-day count
  • Lag / Rolling: configurable lag days, rolling mean/std/min/max
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    ATR_PERIOD,
    BB_STD,
    BB_WINDOW,
    LAG_DAYS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    ROLLING_WINDOWS,
    RSI_PERIOD,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ── Low-level indicator helpers ───────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series,
          fast: int = MACD_FAST,
          slow: int = MACD_SLOW,
          signal: int = MACD_SIGNAL):
    ema_fast   = _ema(series, fast)
    ema_slow   = _ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series,
               window: int = BB_WINDOW,
               n_std:  float = BB_STD):
    mid   = _sma(series, window)
    std   = series.rolling(window, min_periods=1).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = upper - lower
    pct_b = (series - lower) / width.replace(0, np.nan)
    return upper, mid, lower, width, pct_b


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = ATR_PERIOD) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3):
    lowest  = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
    return stoch_k, stoch_d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=1).max()
    lowest  = low.rolling(period, min_periods=1).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change."""
    return series.pct_change(period) * 100


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series,
          volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


def _historical_vol(series: pd.Series, window: int = 21) -> pd.Series:
    """Annualised historical volatility."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window, min_periods=1).std() * np.sqrt(252)


# ── Main feature builder ──────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the full suite of technical indicators to a cleaned OHLCV DataFrame."""
    df = df.copy()
    c = df[TARGET_COL]       # Close
    o = df.get("Open",  c)
    h = df.get("High",  c)
    l = df.get("Low",   c)
    v = df.get("Volume", pd.Series(0, index=df.index))

    # ── Moving Averages ───────────────────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        df[f"SMA_{w}"]     = _sma(c, w)
        df[f"EMA_{w}"]     = _ema(c, w)
        df[f"Price_SMA_{w}_ratio"] = c / df[f"SMA_{w}"].replace(0, np.nan)

    # ── MACD ──────────────────────────────────────────────────────────────────
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = _macd(c)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI"] = _rsi(c, RSI_PERIOD)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"], df["BB_Width"], df["BB_PctB"] = \
        _bollinger(c)

    # ── ATR ───────────────────────────────────────────────────────────────────
    df["ATR"] = _atr(h, l, c)
    df["ATR_ratio"] = df["ATR"] / c  # normalised ATR

    # ── Stochastic ────────────────────────────────────────────────────────────
    df["Stoch_K"], df["Stoch_D"] = _stochastic(h, l, c)

    # ── Williams %R ───────────────────────────────────────────────────────────
    df["Williams_R"] = _williams_r(h, l, c)

    # ── Rate of Change ────────────────────────────────────────────────────────
    df["ROC_5"]  = _roc(c, 5)
    df["ROC_10"] = _roc(c, 10)
    df["ROC_21"] = _roc(c, 21)

    # ── Historical Volatility ─────────────────────────────────────────────────
    df["HV_21"]  = _historical_vol(c, 21)
    df["HV_63"]  = _historical_vol(c, 63)

    # ── Volume indicators ─────────────────────────────────────────────────────
    if v.sum() > 0:
        df["OBV"]           = _obv(c, v)
        df["VWAP"]          = _vwap(h, l, c, v)
        df["Volume_SMA_20"] = _sma(v, 20)
        df["Volume_ratio"]  = v / df["Volume_SMA_20"].replace(0, np.nan)
        df["OBV_EMA"]       = _ema(df["OBV"], 20)

    # ── Price patterns ────────────────────────────────────────────────────────
    df["High_Low_range"]  = h - l
    df["Close_Open_diff"] = c - o
    df["Shadow_Upper"]    = h - pd.concat([c, o], axis=1).max(axis=1)
    df["Shadow_Lower"]    = pd.concat([c, o], axis=1).min(axis=1) - l

    logger.info("Technical indicators added: %d features", len(df.columns))
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lag_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add lag features for the target column."""
    if lag_days is None:
        lag_days = LAG_DAYS
    df = df.copy()
    for lag in lag_days:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Rolling mean, std, min, max, skew, kurt for the target column."""
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy()
    for w in windows:
        r = df[target_col].rolling(w, min_periods=1)
        df[f"{target_col}_roll_mean_{w}"] = r.mean()
        df[f"{target_col}_roll_std_{w}"]  = r.std()
        df[f"{target_col}_roll_min_{w}"]  = r.min()
        df[f"{target_col}_roll_max_{w}"]  = r.max()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode temporal information the model can use for seasonality:
      day-of-week, month, quarter, is-month-end, trading day of year.
    """
    df = df.copy()
    idx = df.index
    df["DayOfWeek"]    = idx.dayofweek                      # 0=Mon … 4=Fri
    df["Month"]        = idx.month
    df["Quarter"]      = idx.quarter
    df["IsMonthEnd"]   = idx.is_month_end.astype(int)
    df["IsMonthStart"] = idx.is_month_start.astype(int)
    df["IsQtrEnd"]     = idx.is_quarter_end.astype(int)
    df["WeekOfYear"]   = idx.isocalendar().week.astype(int)
    df["DayOfYear"]    = idx.dayofyear

    # Cyclical encoding (sin/cos) so model understands circularity
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 5)
    df["Month_sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]     = np.cos(2 * np.pi * df["Month"] / 12)

    return df


def add_scale_free_features(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Price-level-independent versions of the indicators.

    NVIDIA traded below $6 for the whole training period and at $15-$190 in the
    test period, so features measured in dollars (SMA, lags, ATR, band levels)
    fall completely outside the range the models were trained on. Returns,
    ratios and z-scores mean the same thing at $1 or $190, which lets the ML
    models generalise across NVIDIA's growth.
    """
    df = df.copy()
    c = df[target_col]
    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(c / c.shift(1))
    if "Return" not in df.columns:
        df["Return"] = c.pct_change()

    for lag in LAG_DAYS:
        df[f"LogRet_{lag}d"] = np.log(c / c.shift(lag))          # cumulative return
    for lag in [1, 2, 3, 5]:
        df[f"Log_Return_lag_{lag}"] = df["Log_Return"].shift(lag)  # past daily returns
    for w in ROLLING_WINDOWS:
        r = c.rolling(w, min_periods=2)
        df[f"Close_zscore_{w}"] = (c - r.mean()) / r.std().replace(0, np.nan)
        df[f"Close_to_max_{w}"] = c / r.max() - 1
        df[f"Close_to_min_{w}"] = c / r.min() - 1
    if "MACD" in df.columns:
        df["MACD_norm"]        = df["MACD"] / c
        df["MACD_Signal_norm"] = df["MACD_Signal"] / c
        df["MACD_Hist_norm"]   = df["MACD_Hist"] / c
    if "BB_Width" in df.columns:
        df["BB_Width_norm"] = df["BB_Width"] / df["BB_Mid"].replace(0, np.nan)
    if {"High", "Low", "Open"}.issubset(df.columns):
        df["HL_range_pct"]     = (df["High"] - df["Low"]) / c
        df["CO_pct"]           = (c - df["Open"]) / df["Open"].replace(0, np.nan)
        body_hi = pd.concat([c, df["Open"]], axis=1).max(axis=1)
        body_lo = pd.concat([c, df["Open"]], axis=1).min(axis=1)
        df["Shadow_Upper_pct"] = (df["High"] - body_hi) / c
        df["Shadow_Lower_pct"] = (body_lo - df["Low"]) / c
    if {"OBV", "OBV_EMA", "Volume_SMA_20"}.issubset(df.columns):
        df["OBV_trend"] = (df["OBV"] - df["OBV_EMA"]) / df["Volume_SMA_20"].replace(0, np.nan)
    return df


# Columns used as model inputs by XGBoost and the LSTM (all scale-free).
_MODEL_FEATURE_PREFIXES = (
    "Return", "Log_Return", "LogRet_", "Price_SMA_", "Close_zscore_",
    "Close_to_max_", "Close_to_min_", "MACD_norm", "MACD_Signal_norm",
    "MACD_Hist_norm", "RSI", "BB_PctB", "BB_Width_norm", "ATR_ratio",
    "Stoch_", "Williams_R", "ROC_", "HV_", "Volume_ratio", "OBV_trend",
    "HL_range_pct", "CO_pct", "Shadow_Upper_pct", "Shadow_Lower_pct",
    "DayOfWeek", "Month_sin", "Month_cos", "IsMonthEnd", "IsMonthStart",
    "IsQtrEnd",
)


def get_model_feature_columns(df: pd.DataFrame) -> List[str]:
    """Scale-free feature columns used as model inputs (order is stable)."""
    return [c for c in df.columns
            if c.startswith(_MODEL_FEATURE_PREFIXES) and df[c].dtype != object]


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline: apply all feature engineering steps in correct order.
    Returns a DataFrame with all features, NaN rows dropped.
    """
    df = add_technical_indicators(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)
    df = add_calendar_features(df)
    df = add_scale_free_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna()
    logger.info("Feature set built: %d rows (dropped %d NaN rows), %d features",
                len(df), n_before - len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Return sorted list of feature columns (excludes 'Source' and specified cols)."""
    if exclude is None:
        exclude = ["Source", "Dividends", "Stock Splits"]
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]
