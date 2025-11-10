
# data.py

from __future__ import annotations
import os
import pandas as pd
import yfinance as yf
from typing import Optional
from utils import get_logger

logger = get_logger("data")


def download_data(ticker: str, start: str, end: str, out_path: str) -> str:
    """
    Download daily OHLCV data from Yahoo Finance and save to CSV.
    Returns the written CSV path.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'BIMBOA.MX').
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    out_path : str
        Output CSV path (e.g., 'data/raw/BIMBOA.MX.csv').
    """
    logger.info(f"Downloading {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned by Yahoo Finance for {ticker}.")
    df = df.rename(columns=str.title)  # Ensure columns like 'Open', 'High', ...
    df.index.name = "Date"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    logger.info(f"Saved raw data to {out_path} ({len(df)} rows)")
    return out_path


def load_price_data(path: str) -> pd.DataFrame:
    """
    Load price CSV, parse Date index, and drop rows with any NaN.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with OHLCV columns.
    """
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    # Drop obviously bad rows; most indicators require full rows.
    df = df.dropna()
    # Ensure required columns exist
    req = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    return df
