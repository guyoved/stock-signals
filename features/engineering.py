"""
Feature engineering – technical indicators (using free 'ta' library).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        logger.warning("Not enough data for features")
        return df

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend / Moving averages
    df["SMA_10"] = ta.trend.sma_indicator(close, window=10)
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
    df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
    df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
    df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

    # MACD
    df["MACD_12_26_9"] = ta.trend.macd(close)
    df["MACDh_12_26_9"] = ta.trend.macd_diff(close)
    df["MACDs_12_26_9"] = ta.trend.macd_signal(close)

    # RSI
    df["RSI_14"] = ta.momentum.rsi(close, window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BBL_20_2.0"] = bb.bollinger_lband()
    df["BBM_20_2.0"] = bb.bollinger_mavg()
    df["BBU_20_2.0"] = bb.bollinger_hband()
    df["BBB_20_2.0"] = bb.bollinger_wband()
    df["BBP_20_2.0"] = bb.bollinger_pband()

    # ATR
    df["ATR_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # ADX
    df["ADX_14"] = ta.trend.adx(high, low, close, window=14)
    df["DMP_14"] = ta.trend.adx_pos(high, low, close, window=14)
    df["DMN_14"] = ta.trend.adx_neg(high, low, close, window=14)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["STOCHk_14_3_3"] = stoch.stoch()
    df["STOCHd_14_3_3"] = stoch.stoch_signal()

    # Volume
    df["Volume_SMA_20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / (df["Volume_SMA_20"] + 1e-9)

    # Price relative to MAs
    df["Close_vs_SMA20"] = close / (df["SMA_20"] + 1e-9) - 1
    df["Close_vs_SMA50"] = close / (df["SMA_50"] + 1e-9) - 1
    df["SMA20_vs_SMA50"] = df["SMA_20"] / (df["SMA_50"] + 1e-9) - 1

    # Momentum
    df["ROC_10"] = ta.momentum.roc(close, window=10)
    df["ROC_20"] = ta.momentum.roc(close, window=20)

    # Returns
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()

    # Target
    df["Target"] = (close.shift(-1) > close).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
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
    return [c for c in preferred if c in df.columns]


def prepare_ml_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = get_feature_columns(df)
    data = df[feature_cols + ["Target"]].dropna()
    X = data[feature_cols]
    y = data["Target"]
    return X, y