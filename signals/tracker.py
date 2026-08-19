"""
Permanent signal tracking using CSV (works with GitHub + Streamlit).
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from config.settings import BASE_DIR, SIGNAL_HORIZON_DAYS

logger = logging.getLogger(__name__)

HISTORY_FILE = BASE_DIR / "data" / "signals_history.csv"


def init_db() -> None:
    """Create the CSV file if it does not exist."""
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    if not HISTORY_FILE.exists():
        df = pd.DataFrame(columns=[
            "id", "timestamp", "ticker", "signal", "confidence", "entry_price",
            "horizon_days", "reason", "status", "exit_price", "exit_date",
            "return_pct", "success", "evaluated_at"
        ])
        df.to_csv(HISTORY_FILE, index=False)
    logger.info(f"Tracking file ready at {HISTORY_FILE}")


def _load_history() -> pd.DataFrame:
    init_db()
    try:
        df = pd.read_csv(HISTORY_FILE)
        return df
    except Exception:
        return pd.DataFrame()


def _save_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_FILE, index=False)


def log_signal(sig: Dict[str, Any], horizon_days: int = SIGNAL_HORIZON_DAYS) -> int:
    init_db()
    df = _load_history()

    new_id = 1 if df.empty else int(df["id"].max()) + 1

    new_row = {
        "id": new_id,
        "timestamp": sig.get("timestamp") or datetime.utcnow().isoformat(),
        "ticker": sig["ticker"],
        "signal": sig["signal"],
        "confidence": sig.get("confidence"),
        "entry_price": sig.get("price"),
        "horizon_days": horizon_days,
        "reason": sig.get("reason"),
        "status": "open",
        "exit_price": None,
        "exit_date": None,
        "return_pct": None,
        "success": None,
        "evaluated_at": None,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_history(df)
    return new_id


def log_signals(signals: List[Dict[str, Any]], horizon_days: int = SIGNAL_HORIZON_DAYS) -> int:
    count = 0
    for s in signals:
        if s.get("signal") in ("BUY", "SELL"):
            log_signal(s, horizon_days=horizon_days)
            count += 1
    return count


def _get_price_on_or_after(ticker: str, date_str: str) -> Optional[float]:
    try:
        start = datetime.fromisoformat(date_str[:10])
        end = start + timedelta(days=12)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].iloc[0])
    except Exception as e:
        logger.warning(f"Could not fetch exit price for {ticker}: {e}")
        return None


def evaluate_open_signals(horizon_days: int = SIGNAL_HORIZON_DAYS) -> int:
    init_db()
    df = _load_history()
    if df.empty:
        return 0

    now = datetime.utcnow()
    evaluated = 0

    for idx, row in df[df["status"] == "open"].iterrows():
        try:
            entry_time = datetime.fromisoformat(str(row["timestamp"])[:19])
            days_passed = (now - entry_time).days

            if days_passed < int(row["horizon_days"]):
                continue

            exit_price = _get_price_on_or_after(row["ticker"], str(row["timestamp"]))
            if exit_price is None or pd.isna(row["entry_price"]):
                continue

            entry = float(row["entry_price"])
            if row["signal"] == "BUY":
                ret = (exit_price / entry) - 1
            else:
                ret = (entry / exit_price) - 1

            success = 1 if ret > 0 else 0

            df.at[idx, "status"] = "closed"
            df.at[idx, "exit_price"] = round(exit_price, 2)
            df.at[idx, "exit_date"] = now.isoformat()
            df.at[idx, "return_pct"] = round(ret * 100, 3)
            df.at[idx, "success"] = success
            df.at[idx, "evaluated_at"] = now.isoformat()
            evaluated += 1
        except Exception as e:
            logger.error(f"Error evaluating signal {row.get('id')}: {e}")

    if evaluated > 0:
        _save_history(df)

    logger.info(f"Evaluated {evaluated} signals")
    return evaluated


def get_performance_stats() -> Dict[str, Any]:
    df = _load_history()
    closed = df[df["status"] == "closed"] if not df.empty else pd.DataFrame()

    if closed.empty:
        return {
            "n_closed": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "total_return_pct": None,
            "n_wins": 0,
            "n_losses": 0,
        }

    wins = closed[closed["success"] == 1]
    losses = closed[closed["success"] == 0]

    return {
        "n_closed": len(closed),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_return_pct": round(closed["return_pct"].mean(), 2),
        "total_return_pct": round(closed["return_pct"].sum(), 2),
        "avg_win_pct": round(wins["return_pct"].mean(), 2) if len(wins) > 0 else None,
        "avg_loss_pct": round(losses["return_pct"].mean(), 2) if len(losses) > 0 else None,
    }


def get_recent_signals(limit: int = 50) -> pd.DataFrame:
    df = _load_history()
    if df.empty:
        return df
    return df.sort_values("timestamp", ascending=False).head(limit)