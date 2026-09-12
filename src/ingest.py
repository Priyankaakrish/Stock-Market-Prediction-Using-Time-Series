"""Stage 1-2: acquire raw NVDA OHLCV data.

Primary source is the Yahoo Finance API via ``yfinance``. Because the pipeline
must be reproducible offline (CI, air-gapped training boxes, graders without a
network), a previously downloaded snapshot on disk is used as a fallback.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from .config import DATA, PATHS

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


class DataValidationError(RuntimeError):
    """Raised when the ingested frame fails a contract check."""


def _download(ticker: str, start: str, end: str | None, retries: int = 3) -> pd.DataFrame:
    """Pull daily OHLCV from Yahoo Finance with exponential back-off."""
    import yfinance as yf  # imported lazily so the module works without network

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end, interval="1d",
                auto_adjust=False, progress=False, threads=False,
            )
            if df is None or df.empty:
                raise DataValidationError(f"yfinance returned no rows for {ticker}")
            if isinstance(df.columns, pd.MultiIndex):     # yfinance >= 0.2.51
                df.columns = df.columns.get_level_values(0)
            return df.reset_index()
        except Exception as err:
            last_err = err
            wait = 2 ** attempt
            log.warning("Download attempt %d/%d failed (%s); retrying in %ss",
                        attempt, retries, err, wait)
            time.sleep(wait)
    raise DataValidationError(f"All {retries} download attempts failed") from last_err


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Contract checks that must hold before anything downstream runs."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise DataValidationError("Unparseable dates present in the Date column")

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if (df[numeric[:-1]] <= 0).any().any():
        raise DataValidationError("Non-positive prices found")

    bad_range = df["High"] < df["Low"]
    if bad_range.any():
        raise DataValidationError(f"{int(bad_range.sum())} rows where High < Low")

    if len(df) < 500:
        raise DataValidationError(f"Only {len(df)} rows ingested; expected a long history")

    return df


def ingest(use_cache: bool = True, force_download: bool = False) -> pd.DataFrame:
    """Return the raw NVDA frame and persist it to ``data/raw/nvda_raw.csv``."""
    PATHS.ensure()

    if use_cache and PATHS.raw.exists() and not force_download:
        log.info("Loading cached raw data from %s", PATHS.raw)
        df = pd.read_csv(PATHS.raw)
    else:
        log.info("Downloading %s from Yahoo Finance", DATA.ticker)
        try:
            df = _download(DATA.ticker, DATA.start, DATA.end)
        except Exception as err:
            if PATHS.raw.exists():
                log.error("Download failed (%s); falling back to cached snapshot", err)
                df = pd.read_csv(PATHS.raw)
            else:
                raise

    df = validate(df)
    df.to_csv(PATHS.raw, index=False)
    log.info("Ingested %d rows spanning %s to %s",
             len(df), df["Date"].min().date(), df["Date"].max().date())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ingest()
