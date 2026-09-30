"""
pipelines/update_data.py
────────────────────────
Download the latest NVIDIA (NVDA) daily prices from Yahoo Finance and write
them in the format the loader expects:

    data/raw/nvda_yahoo_finance.csv
    data/raw/nvda_master_dataset.csv

Run (needs internet access):
    cd nvda_stock_prediction
    python pipelines/update_data.py
    python pipelines/train_pipeline.py        # retrain on the fresh data
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg


def main() -> int:
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Run: pip install yfinance")
        return 1

    hist = yf.Ticker(cfg.TICKER).history(period="max", auto_adjust=False)
    if hist.empty:
        print("No data returned from Yahoo Finance — check your connection.")
        return 1

    hist = hist.reset_index()
    hist["Date"] = hist["Date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    out = hist[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    out["Dividends"] = hist.get("Dividends", 0.0)
    out["Stock Splits"] = hist.get("Stock Splits", 0.0)
    out["Source"] = "Yahoo Finance"

    for key in ("yahoo", "master"):
        path = cfg.DATA_FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        print(f"Saved {len(out)} rows ({out.Date.iloc[0]} .. {out.Date.iloc[-1]}) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
