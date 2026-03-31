"""
data_collection.py
------------------
Downloads and cleans historical OHLCV data for a list of tickers
using the yfinance library.
"""

import yfinance as yf
import pandas as pd


TICKERS = ["SPY", "QQQ", "XLK"]
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"


def download_stock_data(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date (timezone-naive).
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Keep only OHLCV columns
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Flatten MultiIndex columns if present (yfinance >=0.2 may return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove timezone from index
    df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Ensure chronological order
    df.sort_index(inplace=True)

    print(f"[{ticker}] Downloaded {len(df)} trading days ({df.index[0].date()} – {df.index[-1].date()})")
    return df


def download_all(tickers: list = TICKERS, start: str = START_DATE, end: str = END_DATE) -> dict:
    """Download data for all tickers. Returns {ticker: DataFrame}."""
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = download_stock_data(ticker, start, end)
        except Exception as e:
            print(f"[WARNING] Failed to download {ticker}: {e}")
    return data
