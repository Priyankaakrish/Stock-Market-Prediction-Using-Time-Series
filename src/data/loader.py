"""
src/data/loader.py
──────────────────
Multi-source NVIDIA stock data loader.

Handles:
  • Barchart / Investing.com / MarketWatch / NASDAQ / Yahoo Finance CSVs
  • Timezone normalisation (UTC -> date-only index)
  • Duplicate-date deduplication with configurable priority ordering
  • Column standardisation across heterogeneous source formats
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import DATA_FILES, DATA_RAW_DIR, TARGET_COL

logger = logging.getLogger(__name__)


# ── Column name aliases ────────────────────────────────────────────────────────
_COL_MAP = {
    "date":   "Date",
    "open":   "Open",
    "high":   "High",
    "low":    "Low",
    "close":  "Close",
    "volume": "Volume",
}

# Source priority for deduplication (lower index = higher trust)
SOURCE_PRIORITY = ["Yahoo Finance", "Investing.com (via proxy)",
                   "Barchart (via proxy)", "MarketWatch", "NASDAQ"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols / commas and cast to float."""
    if series.dtype == object:
        series = series.str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(series, errors="coerce")


def _parse_date(series: pd.Series) -> pd.Series:
    """Parse heterogeneous date strings -> tz-naive date index."""
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    # Drop timezone, keep date only
    return parsed.dt.tz_localize(None).dt.normalize()


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case then remap column names to the canonical set."""
    df.columns = [c.strip() for c in df.columns]
    rename = {c: _COL_MAP[c.lower()] for c in df.columns if c.lower() in _COL_MAP}
    df = df.rename(columns=rename)
    return df


def _load_single_source(path: Path) -> Optional[pd.DataFrame]:
    """
    Load one source CSV and return a clean DataFrame indexed by Date.
    Returns None if the file is missing or empty.
    """
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return None

    if df.empty:
        logger.warning("Empty file: %s", path)
        return None

    df = _standardise_columns(df)

    # ── Date ──────────────────────────────────────────────────────────────────
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        logger.error("No 'Date' column in %s", path)
        return None
    df["Date"] = _parse_date(df[date_col])
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="first")]

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    # Keep only canonical columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Source"]
            if c in df.columns]
    df = df[keep]

    # Attach source label if missing
    if "Source" not in df.columns:
        df["Source"] = path.stem

    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_all_sources(data_files: Optional[Dict[str, Path]] = None) -> pd.DataFrame:
    """
    Load and concatenate every configured data source.

    Duplicate dates are resolved by SOURCE_PRIORITY order:
    the row from the highest-priority source is retained.
    """
    if data_files is None:
        data_files = DATA_FILES

    frames: List[pd.DataFrame] = []
    for name, path in data_files.items():
        df = _load_single_source(path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data loaded — check DATA_FILES paths in config.py")

    combined = pd.concat(frames)
    combined = combined.sort_index()

    # ── Deduplicate ───────────────────────────────────────────────────────────
    if "Source" in combined.columns:
        # Map each source to its priority rank
        priority_map = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
        combined["_priority"] = combined["Source"].map(
            lambda s: next((v for k, v in priority_map.items() if k in str(s)),
                           len(SOURCE_PRIORITY))
        )
        combined = combined.sort_values("_priority", kind="stable")
        combined = (combined[~combined.index.duplicated(keep="first")]
                    .drop(columns=["_priority"])
                    .sort_index())
    else:
        combined = combined.loc[~combined.index.duplicated(keep="first")]

    # ── Sanity checks ─────────────────────────────────────────────────────────
    n_missing = combined[TARGET_COL].isna().sum()
    if n_missing > 0:
        logger.warning("Missing values in %s: %d rows — forward-filling",
                       TARGET_COL, n_missing)
        combined[TARGET_COL] = combined[TARGET_COL].ffill()

    logger.info("Combined dataset: %d trading days (%s -> %s)",
                len(combined),
                combined.index.min().date(),
                combined.index.max().date())
    return combined


def load_master_only() -> pd.DataFrame:
    """Shortcut: load only the pre-merged master CSV."""
    path = DATA_FILES["master"]
    df = _load_single_source(path)
    if df is None:
        raise FileNotFoundError(f"Master dataset not found at {path}")
    return df


def load_nasdaq_recent() -> pd.DataFrame:
    """
    Load the NASDAQ CSV which contains the most recent prices
    (distinct format: MM/DD/YYYY dates, $ prefixed close).
    """
    path = DATA_FILES.get("nasdaq", DATA_RAW_DIR / "nvda_nasdaq.csv")
    df = _load_single_source(path)
    if df is None:
        raise FileNotFoundError(f"NASDAQ file not found at {path}")
    return df
