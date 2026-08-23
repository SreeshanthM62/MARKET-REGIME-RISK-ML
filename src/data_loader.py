from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/raw")

TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "TLT": "TLT",
    "GLD": "GLD",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "DXY": "DX-Y.NYB",
}


def _download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data and return a normalized single-level DataFrame."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    df = df[required].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def download_market_data(
    start: str = "2015-01-01",
    end: str = "2026-01-01",
    output_dir: Path = DATA_DIR,
) -> None:
    """Download the research universe from Yahoo Finance and save one CSV per ticker."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        df = _download_one(ticker, start, end)
        df.to_csv(output_dir / f"{name.lower()}.csv")
        print(f"  {len(df):,} rows -> {output_dir / f'{name.lower()}.csv'}")


def load_raw_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load previously downloaded CSV files."""
    data = {}
    for name in TICKERS:
        path = data_dir / f"{name.lower()}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python scripts/download_data.py` first."
            )
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        data[name] = df.sort_index()
    return data
