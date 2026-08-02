"""
Simple vectorized-ish backtester for the signal logic.
Free, transparent, good enough for initial validation.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from features.engineering import add_technical_features
from models.trainer import train_lightgbm, predict_proba, get_signal_from_proba

logger = logging.getLogger(__name__)


def run_simple_backtest(
    df: pd.DataFrame,
    min_confidence: float = 0.58,
    holding_period: int = 5,
    transaction_cost: float = 0.001,
) -> Dict:
    """
    Very simple backtest:
    - Train on first 70% of data
    - Generate signals on the remaining 30%
    - Enter on signal, exit after `holding_period` days or opposite signal
    - Account for transaction costs
    """
    if len(df) < 200:
        return {"error": "Not enough data"}

    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    # Train
    model, metrics = train_lightgbm(train_df)
    model_data = {"model": model, "features": list(model.feature_name_)}

    # Generate signals on test set (rolling style – simplified)
    test_feat = add_technical_features(test_df)
    feature_names = model_data["features"]

    positions = []
    equity = [1.0]
    returns = []

    i = 50  # warm-up
    while i < len(test_feat) - holding_period:
        window = test_feat.iloc[: i + 1]
        proba = predict_proba(model_data, window)
        signal = get_signal_from_proba(proba, min_confidence) if proba else "HOLD"

        if signal == "BUY":
            entry_price = test_feat["Close"].iloc[i]
            exit_price = test_feat["Close"].iloc[i + holding_period]
            ret = (exit_price / entry_price - 1) - 2 * transaction_cost
            returns.append(ret)
            equity.append(equity[-1] * (1 + ret))
            positions.append({"entry_idx": i, "side": "BUY", "ret": ret})
            i += holding_period
        elif signal == "SELL":
            entry_price = test_feat["Close"].iloc[i]
            exit_price = test_feat["Close"].iloc[i + holding_period]
            ret = (entry_price / exit_price - 1) - 2 * transaction_cost  # short
            returns.append(ret)
            equity.append(equity[-1] * (1 + ret))
            positions.append({"entry_idx": i, "side": "SELL", "ret": ret})
            i += holding_period
        else:
            i += 1

    if not returns:
        return {"error": "No trades generated", "model_metrics": metrics}

    rets = np.array(returns)
    total_return = equity[-1] - 1
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252 / holding_period)

    return {
        "total_return": float(total_return),
        "n_trades": len(returns),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(abs(avg_win / avg_loss)) if avg_loss != 0 else None,
        "sharpe_approx": float(sharpe),
        "final_equity": float(equity[-1]),
        "model_metrics": metrics,
    }
