"""
src/forecasting.py
──────────────────
Shared forecasting logic used by the training pipeline, the batch predict
pipeline and the API, so all three produce forecasts the same way.

Design (why it differs from a plain "predict the price" setup)
---------------------------------------------------------------
NVIDIA's split-adjusted price went from ~$0.04 (1999) to ~$190 (2026). With a
70/15/15 time split, every training price is below ~$6.2 while the test period
runs from ~$11 to ~$207. Models that predict the *price level* from
*price-level* features cannot extrapolate that far (tree models literally cap
at the largest training value). So:

  • XGBoost and the LSTM predict the NEXT-DAY LOG RETURN from scale-free
    features (returns, ratios, oscillators, z-scores) and the price is
    rebuilt as  close_t × exp(predicted return).
  • ARIMA and Prophet are fitted on log(price).
  • Every model is evaluated one step ahead: the prediction for day t uses
    information up to day t-1 only (no look-ahead).
  • Multi-day forecasts are produced recursively: predict tomorrow, append it
    to the price history, recompute the features, repeat.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import FORECAST_DAYS, MODELS_DIR, TARGET_COL
from src.data.preprocessor import add_returns
from src.features.engineer import build_feature_set

logger = logging.getLogger(__name__)

TARGET_RET = "Target_LogRet"            # log(close_{t+1} / close_t)
LSTM_TARGET_SCALE = 100.0               # LSTM learns returns in percent
FEATURES_FILE = MODELS_DIR / "model_features.json"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


# ── Feature frames & targets ──────────────────────────────────────────────────

def make_model_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the full feature matrix from an OHLCV frame."""
    base = ohlcv[[c for c in OHLCV if c in ohlcv.columns]].copy()
    base = add_returns(base)
    return build_feature_set(base)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Next-day log return target (NaN on the last row)."""
    df = df.copy()
    df[TARGET_RET] = np.log(df[TARGET_COL].shift(-1) / df[TARGET_COL])
    return df


def save_feature_list(cols: List[str], path: Path = FEATURES_FILE) -> None:
    with open(path, "w") as f:
        json.dump(cols, f, indent=2)


def load_feature_list(path: Path = FEATURES_FILE) -> List[str]:
    with open(path) as f:
        return json.load(f)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def xgb_logret(gb, frame: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    """Predicted next-day log return for every row of `frame`."""
    return np.asarray(gb.predict(frame[feat_cols].values), dtype=float)


# ── LSTM ──────────────────────────────────────────────────────────────────────

def make_windows(
    feats: np.ndarray,
    target: Optional[np.ndarray],
    lookback: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Sliding windows that END at row t (inclusive) -> target of row t.
    Returns X (n, lookback, f), y (n, 1) or None, and the end-row indices.
    """
    ends = np.arange(lookback - 1, len(feats))
    X = np.stack([feats[e - lookback + 1: e + 1] for e in ends]).astype("float32") \
        if len(ends) else np.empty((0, lookback, feats.shape[1]), "float32")
    y = None
    if target is not None:
        y = (target[ends] * LSTM_TARGET_SCALE).reshape(-1, 1).astype("float32")
    return X, y, ends


def lstm_logret(lf, frame: pd.DataFrame, feat_cols: List[str], scaler) -> np.ndarray:
    """Predicted next-day log return for every row (NaN before the first full window)."""
    sc = scaler.transform(frame, feat_cols)[feat_cols].values.astype("float32")
    sc = np.nan_to_num(np.clip(sc, -10, 10))
    X, _, ends = make_windows(sc, None, lf.lookback)
    out = np.full(len(frame), np.nan)
    if len(X):
        out[ends] = lf.predict(X).flatten() / LSTM_TARGET_SCALE
    return out


# ── One-step evaluation helpers ───────────────────────────────────────────────

def returns_to_next_prices(frame: pd.DataFrame, logret: np.ndarray) -> pd.Series:
    """
    Convert per-row predicted log returns into price predictions indexed by
    the day being predicted (the next row's date).
    """
    prices = frame[TARGET_COL].values * np.exp(logret)
    return pd.Series(prices[:-1], index=frame.index[1:])


# ── Recursive multi-day forecasting ──────────────────────────────────────────

def recursive_forecast(
    ohlcv: pd.DataFrame,
    next_logret: Callable[[pd.DataFrame], float],
    steps: int = FORECAST_DAYS,
    history_rows: int = 400,
) -> np.ndarray:
    """
    Iteratively forecast `steps` business days of closes.

    next_logret(frame) must return the predicted log return for the day after
    the frame's last row. Each prediction is appended as a synthetic bar
    (O=H=L=C=predicted close, volume = recent average) and the features are
    recomputed before the next step.
    """
    hist = ohlcv[[c for c in OHLCV if c in ohlcv.columns]].tail(history_rows).copy()
    avg_vol = float(hist["Volume"].tail(20).mean()) if "Volume" in hist else 0.0
    closes = []
    for _ in range(steps):
        frame = make_model_frame(hist)
        r = float(next_logret(frame))
        if not np.isfinite(r):
            r = 0.0
        nxt = float(hist[TARGET_COL].iloc[-1]) * float(np.exp(r))
        day = hist.index[-1] + pd.offsets.BDay(1)
        hist.loc[day] = {"Open": nxt, "High": nxt, "Low": nxt,
                         "Close": nxt, "Volume": avg_vol}
        closes.append(nxt)
    return np.array(closes)


def forecast_all(
    models: Dict[str, object],
    ohlcv: pd.DataFrame,
    steps: int = FORECAST_DAYS,
    scaler=None,
    feat_cols: Optional[List[str]] = None,
    which: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Forward forecast for every available model.
    Returns {model_name: array of `steps` predicted closes}.
    """
    which = which or list(models.keys())
    last_date = ohlcv.index[-1]
    last_close = float(ohlcv[TARGET_COL].iloc[-1])
    out: Dict[str, np.ndarray] = {}

    for name in which:
        if name not in models:
            continue
        m = models[name]
        try:
            if name == "arima":
                preds, _ = m.forecast(steps=steps)
                out[name] = np.asarray(preds, dtype=float).flatten()
            elif name == "prophet":
                _, vals, _, _ = m.forecast_path(last_date, last_close, steps)
                out[name] = np.asarray(vals, dtype=float)
            elif name == "xgboost":
                out[name] = recursive_forecast(
                    ohlcv, lambda f: xgb_logret(m, f.iloc[[-1]], feat_cols)[0], steps)
            elif name == "lstm":
                tail = m.lookback + 5
                out[name] = recursive_forecast(
                    ohlcv, lambda f: lstm_logret(m, f.tail(tail), feat_cols, scaler)[-1],
                    steps)
        except Exception as exc:          # one bad model must not kill the rest
            logger.warning("%s forecast failed: %s", name, exc)
    return out
