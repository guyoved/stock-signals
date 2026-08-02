"""
Feature engineering – technical indicators (all free).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a rich set of technical features to OHLCV DataFrame.
    Expects columns: Open, High, Low, Close, Volume
    """
    if df.empty or len(df) < 50:
        logger.warning("Not enough data for features")
        return df

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend / Moving averages
    df["SMA_10"] = ta.sma(close, length=10)
    df["SMA_20"] = ta.sma(close, length=20)
    df["SMA_50"] = ta.sma(close, length=50)
    df["SMA_200"] = ta.sma(close, length=200)
    df["EMA_12"] = ta.ema(close, length=12)
    df["EMA_26"] = ta.ema(close, length=26)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    # RSI
    df["RSI_14"] = ta.rsi(close, length=14)

    # Bollinger Bands
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)

    # ATR (volatility)
    df["ATR_14"] = ta.atr(high, low, close, length=14)

    # ADX (trend strength)
    adx = ta.adx(high, low, close, length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    # Stochastic
    stoch = ta.stoch(high, low, close)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)

    # Volume features
    df["Volume_SMA_20"] = ta.sma(volume, length=20)
    df["Volume_Ratio"] = volume / (df["Volume_SMA_20"] + 1e-9)

    # Price relative to MAs
    df["Close_vs_SMA20"] = close / (df["SMA_20"] + 1e-9) - 1
    df["Close_vs_SMA50"] = close / (df["SMA_50"] + 1e-9) - 1
    df["SMA20_vs_SMA50"] = df["SMA_20"] / (df["SMA_50"] + 1e-9) - 1

    # Momentum
    df["ROC_10"] = ta.roc(close, length=10)
    df["ROC_20"] = ta.roc(close, length=20)

    # Returns
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)

    # Volatility
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # Target for supervised learning: next-day direction (1 = up, 0 = down)
    df["Target"] = (close.shift(-1) > close).astype(float)

    # Clean infinite / NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of columns that should be used as model features."""
    exclude = {
        "Open", "High", "Low", "Close", "Volume", "Ticker",
        "Target", "Date"
    }
    # Also exclude raw BB / MACD helper columns if any leftover
    cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("BBL_")  # keep BBM, BBU, BBB etc if useful
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    # Prefer explicit useful ones
    preferred = [
        "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_12", "EMA_26",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "RSI_14",
        "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
        "ATR_14",
        "ADX_14", "DMP_14", "DMN_14",
        "STOCHk_14_3_3", "STOCHd_14_3_3",
        "Volume_Ratio",
        "Close_vs_SMA20", "Close_vs_SMA50", "SMA20_vs_SMA50",
        "ROC_10", "ROC_20",
        "Return_1d", "Return_5d",
        "Volatility_20",
    ]
    # Keep only those that actually exist
    final = [c for c in preferred if c in df.columns]
    # Add any remaining numeric that look useful
    for c in cols:
        if c not in final:
            final.append(c)
    return final


def prepare_ml_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return X, y ready for training.
    Drops rows with NaN in features or target.
    """
    feature_cols = get_feature_columns(df)
    data = df[feature_cols + ["Target"]].dropna()
    X = data[feature_cols]
    y = data["Target"]
    return X, y
