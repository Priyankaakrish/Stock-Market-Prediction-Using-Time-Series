"""
src/data/preprocessor.py
─────────────────────────
Data cleaning, outlier removal, train/val/test splitting,
and MinMax / Standard scalers for LSTM and tabular models.

Industry pattern:
  raw DataFrame -> clean -> engineer (see features/) -> split -> scale
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import (
    DATA_PROC_DIR,
    MODELS_DIR,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


# ── Cleaning ──────────────────────────────────────────────────────────────────

def remove_outliers_iqr(
    df: pd.DataFrame,
    cols: Optional[list] = None,
    factor: float = 3.0,
    window: int = 21,
) -> pd.DataFrame:
    """
    Remove bad ticks without destroying the long-run trend.

    NVIDIA's price rose from ~$0.04 (split-adjusted) to ~$190, so an IQR test
    on raw price *levels* would flag the whole recent rally as "outliers".
    Instead we measure each value's log-distance from its own rolling median
    and apply the IQR rule to that deviation. Genuine trends move the median
    along with the price; isolated spikes/typos stand out and are replaced
    (NaN -> forward-fill).
    """
    if cols is None:
        cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        x = df[col].where(df[col] > 0)
        med = x.rolling(window, center=True, min_periods=1).median()
        dev = np.log(x / med)
        q1, q3 = dev.quantile(0.25), dev.quantile(0.75)
        iqr = q3 - q1
        # floor keeps normal volatile days (e.g. +/-25% earnings moves) intact
        spread = max(factor * iqr, 0.5 if col != "Volume" else 2.0)
        mask = (dev < q1 - spread) | (dev > q3 + spread) | x.isna()
        n_outliers = int(mask.sum())
        if n_outliers > 0:
            logger.info("Outliers in %s: %d rows replaced", col, n_outliers)
            df.loc[mask, col] = np.nan
    df = df.ffill().bfill()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-return and simple return columns."""
    df = df.copy()
    df["Return"]     = df[TARGET_COL].pct_change()
    df["Log_Return"] = np.log(df[TARGET_COL] / df[TARGET_COL].shift(1))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Drop weekends & ensure business-day index
      2. Forward-fill small gaps (≤ 3 days)
      3. Drop rows with all-NaN OHLCV
      4. Remove IQR outliers
      5. Add returns
    """
    df = df.copy()

    # Keep only trading days (Mon-Fri)
    df = df[df.index.dayofweek < 5]

    # Drop duplicate dates (can arise when multiple sources share the same date
    # and the priority-dedup in loader didn't fully resolve all overlaps after
    # timezone stripping normalises slightly different timestamps to the same date)
    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        logger.warning("Dropping %d duplicate date(s) before reindex", n_dups)
        df = df[~df.index.duplicated(keep="first")]

    # Reindex to business day calendar and fill small gaps
    bday_idx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bday_idx)
    df = df.ffill(limit=3)

    # Drop remaining NaN rows
    ohlcv = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df.dropna(subset=[TARGET_COL])

    df = remove_outliers_iqr(df, cols=ohlcv)
    df = add_returns(df)

    logger.info("Clean data: %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Train / Val / Test Split ──────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological (non-shuffled) split preserving temporal order.
    Returns (train, val, test) DataFrames.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train: n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    logger.info("Split -> train: %d | val: %d | test: %d",
                len(train), len(val), len(test))
    return train, val, test


# ── Scalers ───────────────────────────────────────────────────────────────────

class DataScaler:
    """
    Wraps sklearn scalers with fit-on-train-only discipline.
    Supports 'minmax', 'standard', 'robust'.
    """

    def __init__(self, scaler_type: str = "minmax", feature_range: Tuple = (0, 1)):
        self.scaler_type = scaler_type
        self._scalers: Dict[str, object] = {}
        self._feature_range = feature_range

    def _make_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=self._feature_range)
        elif self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def fit_transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Fit on provided data and transform."""
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                continue
            scaler = self._make_scaler()
            df[col] = scaler.fit_transform(df[[col]])
            self._scalers[col] = scaler
        return df

    def transform(self, df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
        """Transform using already-fitted scalers."""
        df = df.copy()
        cols = cols or list(self._scalers.keys())
        for col in cols:
            if col not in df.columns or col not in self._scalers:
                continue
            df[col] = self._scalers[col].transform(df[[col]])
        return df

    def inverse_transform_col(self, values: np.ndarray, col: str) -> np.ndarray:
        """Inverse-transform a single column back to original scale."""
        if col not in self._scalers:
            raise KeyError(f"Scaler for '{col}' not found — call fit_transform first")
        vals = np.array(values).reshape(-1, 1)
        return self._scalers[col].inverse_transform(vals).flatten()

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self._scalers, f)
        logger.info("Scalers saved -> %s", path)

    def load(self, path: Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            self._scalers = pickle.load(f)
        logger.info("Scalers loaded ← %s", path)


# ── Sequence builder (LSTM) ───────────────────────────────────────────────────

def build_sequences(
    arr: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D array (n_samples, n_features) into supervised
    sequence tensors for LSTM training.

    Returns:
        X  shape (n, lookback, n_features)
        y  shape (n, horizon)           — target is first column of arr
    """
    X, y = [], []
    for i in range(lookback, len(arr) - horizon + 1):
        X.append(arr[i - lookback: i])
        y.append(arr[i: i + horizon, 0])   # first column = scaled Close
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_type: str = "minmax",
    lookback:    int  = 60,
    horizon:     int  = 1,
) -> Dict:
    """
    End-to-end preparation:
      1. Temporal split
      2. Fit scaler on train, apply to val/test
      3. Build LSTM sequences for each split

    Returns a dict with keys:
      train_df, val_df, test_df,
      X_train, y_train, X_val, y_val, X_test, y_test,
      scaler
    """
    train_df, val_df, test_df = temporal_split(df)

    scaler = DataScaler(scaler_type=scaler_type)
    cols_to_scale = [c for c in feature_cols if c in train_df.columns]

    train_sc = scaler.fit_transform(train_df, cols_to_scale)
    val_sc   = scaler.transform(val_df,   cols_to_scale)
    test_sc  = scaler.transform(test_df,  cols_to_scale)

    # Reorder so TARGET_COL is always column 0
    ordered_cols = [TARGET_COL] + [c for c in cols_to_scale if c != TARGET_COL]

    def _seq(sc_df):
        arr = sc_df[ordered_cols].values
        return build_sequences(arr, lookback=lookback, horizon=horizon)

    X_tr, y_tr = _seq(train_sc)
    X_va, y_va = _seq(val_sc)
    X_te, y_te = _seq(test_sc)

    # Save scaler
    scaler.save(MODELS_DIR / "feature_scaler.pkl")

    return dict(
        train_df=train_df, val_df=val_df, test_df=test_df,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        scaler=scaler,
        feature_cols=ordered_cols,
    )
