"""
Signal generation engine with Layer 1 quality filters.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import (
    DEFAULT_WATCHLIST,
    MIN_CONFIDENCE,
    SIGNAL_HORIZON_DAYS,
    MAX_SIGNALS_PER_RUN,
    REQUIRE_SPY_TREND,
    REQUIRE_VOLUME_CONFIRM,
    REQUIRE_TREND_FILTER,
    VOLUME_CONFIRM_RATIO,
    TREND_CONFIRM_MARGIN,
    MAX_POSITION_PCT,
)
from data.fetcher import fetch_ohlcv, get_latest_price
from features.engineering import add_technical_features, prepare_ml_dataset
from models.trainer import (
    load_model,
    predict_proba,
    get_signal_from_proba,
    train_lightgbm_from_prepared,
    save_model,
)

logger = logging.getLogger(__name__)


def _normalize_signal_threshold(min_confidence: float) -> float:
    """Ensure the strategy only trades with materially stronger probabilities."""
    return max(float(min_confidence), MIN_CONFIDENCE)


def _apply_quality_filters(
    signal: str,
    proba: float,
    latest: pd.Series,
    spy_regime: str,
    min_confidence: float,
) -> tuple[str, str, list[str], list[str]]:
    """Return a cleaned signal and the reasons for acceptance or rejection."""
    filters_passed: list[str] = []
    filters_failed: list[str] = []
    signal_threshold = _normalize_signal_threshold(min_confidence)

    if signal in ("BUY", "SELL") and proba is not None:
        if signal == "BUY" and proba < signal_threshold:
            signal = "HOLD"
            filters_failed.append(f"Model below threshold ({proba:.1%} < {signal_threshold:.1%})")
        elif signal == "SELL" and proba > (1 - signal_threshold):
            signal = "HOLD"
            filters_failed.append(f"Model below sell threshold ({proba:.1%} > {(1 - signal_threshold):.1%})")

    if REQUIRE_SPY_TREND and signal in ("BUY", "SELL"):
        if signal == "BUY" and spy_regime == "bear":
            signal = "HOLD"
            filters_failed.append("SPY in downtrend")
        elif signal == "SELL" and spy_regime == "bull":
            signal = "HOLD"
            filters_failed.append("SPY in uptrend")
        else:
            filters_passed.append(f"SPY regime OK ({spy_regime})")

    if REQUIRE_VOLUME_CONFIRM and signal in ("BUY", "SELL"):
        vol_ratio = float(latest.get("Volume_Ratio", 1.0))
        if vol_ratio < VOLUME_CONFIRM_RATIO:
            signal = "HOLD"
            filters_failed.append(f"Low volume ({vol_ratio:.2f}x < {VOLUME_CONFIRM_RATIO:.2f}x)")
        else:
            filters_passed.append(f"Volume OK ({vol_ratio:.2f}x)")

    if REQUIRE_TREND_FILTER and signal in ("BUY", "SELL"):
        close_vs_sma20 = float(latest.get("Close_vs_SMA20", 0.0))
        close_vs_sma50 = float(latest.get("Close_vs_SMA50", 0.0))
        sma20_vs_sma50 = float(latest.get("SMA20_vs_SMA50", 0.0))
        rsi = float(latest.get("RSI_14", 50.0))
        roc_10 = float(latest.get("ROC_10", 0.0))

        if signal == "BUY":
            if close_vs_sma20 <= 0.0 or close_vs_sma50 <= 0.0:
                signal = "HOLD"
                filters_failed.append(
                    f"Weak trend (SMA20 {close_vs_sma20:.2%}, SMA50 {close_vs_sma50:.2%})"
                )
            elif sma20_vs_sma50 <= 0.0:
                signal = "HOLD"
                filters_failed.append(f"Short trend not confirming (SMA20/SMA50 {sma20_vs_sma50:.2%})")
            elif rsi < 45 or rsi > 70:
                signal = "HOLD"
                filters_failed.append(f"RSI not confirming ({rsi:.1f})")
            elif roc_10 <= 0.0:
                signal = "HOLD"
                filters_failed.append(f"Momentum weak (ROC10 {roc_10:.2%})")
            else:
                filters_passed.append("Trend + momentum OK")
        elif signal == "SELL":
            if close_vs_sma20 >= 0.0 or close_vs_sma50 >= 0.0:
                signal = "HOLD"
                filters_failed.append(
                    f"Weak downtrend (SMA20 {close_vs_sma20:.2%}, SMA50 {close_vs_sma50:.2%})"
                )
            elif sma20_vs_sma50 >= 0.0:
                signal = "HOLD"
                filters_failed.append(f"Short trend not confirming (SMA20/SMA50 {sma20_vs_sma50:.2%})")
            elif rsi > 55 or rsi < 30:
                signal = "HOLD"
                filters_failed.append(f"RSI not confirming ({rsi:.1f})")
            elif roc_10 >= 0.0:
                signal = "HOLD"
                filters_failed.append(f"Momentum weak (ROC10 {roc_10:.2%})")
            else:
                filters_passed.append("Trend + momentum OK")

    return signal, f"Model probability {proba:.1%}" if proba is not None else "Technical setup", filters_passed, filters_failed


def _get_spy_regime() -> str:
    """Return 'bull', 'bear', or 'neutral' based on SPY vs SMA50."""
    try:
        df = fetch_ohlcv("SPY", period="1y")
        if df.empty or len(df) < 60:
            return "neutral"
        df = add_technical_features(df)
        latest = df.iloc[-1]
        close = latest["Close"]
        sma50 = latest.get("SMA_50", close)
        if close > sma50 * 1.01:
            return "bull"
        if close < sma50 * 0.99:
            return "bear"
        return "neutral"
    except Exception:
        return "neutral"


def generate_signal_for_ticker(
    ticker: str,
    model_data: Optional[dict] = None,
    min_confidence: float = MIN_CONFIDENCE,
    spy_regime: str = "neutral",
) -> Dict[str, Any]:
    df = fetch_ohlcv(ticker, period="2y", interval="1d")
    if df.empty or len(df) < 60:
        return {
            "ticker": ticker,
            "signal": "HOLD",
            "confidence": 0.0,
            "price": None,
            "reason": "Insufficient data",
            "timestamp": datetime.utcnow().isoformat(),
            "atr": None,
            "position_size_pct": None,
        }

    df_feat = add_technical_features(df)
    latest = df_feat.iloc[-1]

    if model_data is None:
        model_data = load_model()

    proba = None
    if model_data is not None:
        proba = predict_proba(model_data, df)

    if proba is None:
        rsi = latest.get("RSI_14", 50)
        close_vs_sma20 = latest.get("Close_vs_SMA20", 0)
        if rsi < 30 and close_vs_sma20 < -0.03:
            signal = "BUY"
            proba = 0.65
            reason = f"RSI oversold ({rsi:.1f}) + below SMA20"
        elif rsi > 70 and close_vs_sma20 > 0.03:
            signal = "SELL"
            proba = 0.35
            reason = f"RSI overbought ({rsi:.1f}) + above SMA20"
        else:
            signal = "HOLD"
            proba = 0.50
            reason = "No strong technical setup"
    else:
        signal = get_signal_from_proba(proba, min_confidence)
        reason = f"Model probability {proba:.1%}"

    # ---------- Layer 1 Filters ----------
    signal, base_reason, filters_passed, filters_failed = _apply_quality_filters(
        signal,
        proba,
        latest,
        spy_regime,
        min_confidence,
    )

    if filters_failed:
        reason = f"{base_reason} | Filtered: {', '.join(filters_failed)}"
    elif filters_passed:
        reason = f"{base_reason} | {', '.join(filters_passed)}"

    price = get_latest_price(ticker) or float(df["Close"].iloc[-1])

    atr = latest.get("ATR_14", None)
    position_size_pct = None
    if atr and price and atr > 0:
        risk_per_share = 2 * atr
        base_position = min(MAX_POSITION_PCT, 0.015 + max(0.0, abs(float(proba) - 0.5)) * 0.10)
        position_size_pct = round(min(MAX_POSITION_PCT, max(0.01, base_position * (1.0 / max(1.0, risk_per_share / price)))), 4)

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(proba, 4) if proba is not None else 0.0,
        "price": round(price, 2) if price else None,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
        "rsi": round(float(latest.get("RSI_14", 0)), 1),
        "atr": round(float(atr), 2) if atr else None,
        "position_size_pct": position_size_pct,
        "horizon_days": SIGNAL_HORIZON_DAYS,
    }


def generate_signals(
    tickers: Optional[List[str]] = None,
    min_confidence: float = MIN_CONFIDENCE,
    only_actionable: bool = True,
    max_signals: int = MAX_SIGNALS_PER_RUN,
) -> List[Dict[str, Any]]:
    tickers = tickers or DEFAULT_WATCHLIST
    model_data = load_model()
    spy_regime = _get_spy_regime()
    logger.info(f"SPY regime: {spy_regime}")

    results = []
    for t in tickers:
        try:
            sig = generate_signal_for_ticker(
                t, model_data=model_data, min_confidence=min_confidence, spy_regime=spy_regime
            )
            if only_actionable and sig["signal"] == "HOLD":
                continue
            results.append(sig)
        except Exception as e:
            logger.error(f"Error generating signal for {t}: {e}")
            continue

    results.sort(key=lambda x: abs(x["confidence"] - 0.5), reverse=True)

    if max_signals and len(results) > max_signals:
        results = results[:max_signals]

    return results


def train_on_watchlist(tickers: Optional[List[str]] = None) -> dict:
    tickers = tickers or DEFAULT_WATCHLIST
    feature_frames = []
    target_frames = []
    for t in tickers:
        df = fetch_ohlcv(t, period="3y")
        if not df.empty and len(df) > 100:
            df = df.reset_index(drop=True)
            df_feat = add_technical_features(df[["Open", "High", "Low", "Close", "Volume"]])
            X, y = prepare_ml_dataset(df_feat)
            if not X.empty:
                feature_frames.append(X)
                target_frames.append(y)

    if not feature_frames:
        raise ValueError("No data fetched for training")

    X = pd.concat(feature_frames, ignore_index=True)
    y = pd.concat(target_frames, ignore_index=True)
    model, metrics = train_lightgbm_from_prepared(X, y)
    save_model(model, metrics)
    return metrics