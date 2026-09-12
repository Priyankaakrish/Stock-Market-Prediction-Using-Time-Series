"""Stage 5: chronological splitting and walk-forward validation windows.

A random ``train_test_split`` on a price series leaks the future into the past
and produces spectacular, meaningless scores. Everything here is strictly
ordered in time.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from .config import DATA, PATHS

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append(
                {
                    "split": name,
                    "rows": len(part),
                    "start": part["Date"].min().date(),
                    "end": part["Date"].max().date(),
                    "close_min": round(float(part["Close"].min()), 2),
                    "close_max": round(float(part["Close"].max()), 2),
                }
            )
        return pd.DataFrame(rows)


def chronological_split(df: pd.DataFrame | None = None, save: bool = True) -> Splits:
    if df is None:
        df = pd.read_csv(PATHS.features)
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)

    n = len(d)
    i_train = int(n * DATA.train_frac)
    i_val = int(n * (DATA.train_frac + DATA.val_frac))

    splits = Splits(
        train=d.iloc[:i_train].reset_index(drop=True),
        val=d.iloc[i_train:i_val].reset_index(drop=True),
        test=d.iloc[i_val:].reset_index(drop=True),
    )

    # Hard guarantee: no timestamp overlap between splits.
    assert splits.train["Date"].max() < splits.val["Date"].min()
    assert splits.val["Date"].max() < splits.test["Date"].min()

    if save:
        PATHS.ensure()
        splits.train.to_csv(PATHS.train, index=False)
        splits.val.to_csv(PATHS.val, index=False)
        splits.test.to_csv(PATHS.test, index=False)

    log.info("Split sizes  train=%d  val=%d  test=%d", len(splits.train),
             len(splits.val), len(splits.test))
    return splits


def walk_forward_windows(
    n: int, initial: int, step: int, horizon: int = 1
) -> Iterator[tuple[slice, slice]]:
    """Yield (train_slice, test_slice) pairs for expanding-window validation."""
    start = initial
    while start + horizon <= n:
        end = min(start + step, n)
        yield slice(0, start), slice(start, end)
        start = end


def load_splits() -> Splits:
    return Splits(
        train=pd.read_csv(PATHS.train, parse_dates=["Date"]),
        val=pd.read_csv(PATHS.val, parse_dates=["Date"]),
        test=pd.read_csv(PATHS.test, parse_dates=["Date"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(chronological_split().describe().to_string(index=False))
