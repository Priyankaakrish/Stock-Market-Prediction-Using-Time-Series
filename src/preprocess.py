"""Stage 3: clean the raw OHLCV frame.

Design note: we deliberately do **not** reindex onto a calendar-day grid.
Stock prices do not exist on weekends and holidays; forward-filling them
invents ~30% fake observations, flattens realised volatility and corrupts
every rolling statistic downstream. The trading-day sequence is the correct
time index for this problem.
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import PATHS

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def preprocess(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(PATHS.raw)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    before = len(df)
    df = df.drop_duplicates(subset="Date", keep="last")
    if len(df) != before:
        log.info("Dropped %d duplicate dates", before - len(df))

    df = df.sort_values("Date").reset_index(drop=True)

    # Gaps inside a price series are missing observations, not missing days:
    # carry the last known price forward, which is what a trader would see.
    n_missing = int(df[OHLCV].isna().sum().sum())
    if n_missing:
        log.info("Forward-filling %d missing OHLCV cells", n_missing)
        df[OHLCV] = df[OHLCV].ffill()
        df = df.dropna(subset=OHLCV).reset_index(drop=True)

    # Zero-volume rows are exchange artefacts (halts); treat as missing.
    zero_vol = df["Volume"] <= 0
    if zero_vol.any():
        log.info("Repairing %d zero-volume rows", int(zero_vol.sum()))
        df.loc[zero_vol, "Volume"] = pd.NA
        df["Volume"] = df["Volume"].ffill().bfill()

    df["Volume"] = df["Volume"].astype("float64")

    # Trading-day gap, useful as a feature and as a data-quality signal.
    df["days_since_prev"] = df["Date"].diff().dt.days.fillna(1.0)

    if save:
        PATHS.ensure()
        df.to_csv(PATHS.cleaned, index=False)
        log.info("Wrote %d cleaned rows -> %s", len(df), PATHS.cleaned)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    preprocess()
