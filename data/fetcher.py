"""
Free market data fetcher using yfinance.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR, LOOKBACK_DAYS

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker.
    Returns DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Flatten multi-index columns if present (newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.title)
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["Ticker"] = ticker
        return df.dropna()
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_multiple(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch data for a list of tickers. Returns dict ticker -> DataFrame."""
    results = {}
    for t in tickers:
        df = fetch_ohlcv(t, period=period, interval=interval)
        if not df.empty:
            results[t] = df
        else:
            logger.warning(f"Skipping {t} – no data")
    return results


def save_parquet(df: pd.DataFrame, ticker: str) -> Path:
    path = DATA_DIR / f"{ticker}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(ticker: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def get_latest_price(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        return float(info.last_price)
    except Exception:
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_ohlcv("AAPL", period="6mo")
    print(df.tail())
    print(f"Shape: {df.shape}")
