"""
Success / Failure tracking of signals.
Stores every signal and later evaluates real performance.
"""
from __future__ import annotations
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from config.settings import DB_PATH, SIGNAL_HORIZON_DAYS

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they do not exist."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL,
            entry_price REAL,
            horizon_days INTEGER,
            reason TEXT,
            status TEXT DEFAULT 'open',
            exit_price REAL,
            exit_date TEXT,
            return_pct REAL,
            success INTEGER,
            evaluated_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Tracking database ready at {DB_PATH}")


def log_signal(sig: Dict[str, Any], horizon_days: int = SIGNAL_HORIZON_DAYS) -> int:
    init_db()
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO signals (
            timestamp, ticker, signal, confidence, entry_price,
            horizon_days, reason, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
    """, (
        sig.get("timestamp") or datetime.utcnow().isoformat(),
        sig["ticker"],
        sig["signal"],
        sig.get("confidence"),
        sig.get("price"),
        horizon_days,
        sig.get("reason"),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


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
        end = start + timedelta(days=10)
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
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM signals WHERE status = 'open'")
    rows = cur.fetchall()
    evaluated = 0
    now = datetime.utcnow()

    for row in rows:
        try:
            entry_time = datetime.fromisoformat(row["timestamp"][:19])
            days_passed = (now - entry_time).days

            if days_passed < row["horizon_days"]:
                continue

            exit_price = _get_price_on_or_after(row["ticker"], row["timestamp"])
            if exit_price is None or row["entry_price"] is None:
                continue

            entry = float(row["entry_price"])
            if row["signal"] == "BUY":
                ret = (exit_price / entry) - 1
            else:
                ret = (entry / exit_price) - 1

            success = 1 if ret > 0 else 0

            cur.execute("""
                UPDATE signals SET
                    status = 'closed',
                    exit_price = ?,
                    exit_date = ?,
                    return_pct = ?,
                    success = ?,
                    evaluated_at = ?
                WHERE id = ?
            """, (
                exit_price,
                now.isoformat(),
                round(ret * 100, 3),
                success,
                now.isoformat(),
                row["id"],
            ))
            evaluated += 1
        except Exception as e:
            logger.error(f"Error evaluating signal {row['id']}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Evaluated {evaluated} signals")
    return evaluated


def get_performance_stats() -> Dict[str, Any]:
    init_db()
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM signals WHERE status = 'closed'", conn)
    conn.close()

    if df.empty:
        return {
            "n_closed": 0,
            "win_rate": None,
            "avg_return_pct": None,
            "total_return_pct": None,
            "n_wins": 0,
            "n_losses": 0,
        }

    wins = df[df["success"] == 1]
    losses = df[df["success"] == 0]

    return {
        "n_closed": len(df),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / len(df) * 100, 1) if len(df) > 0 else None,
        "avg_return_pct": round(df["return_pct"].mean(), 2),
        "total_return_pct": round(df["return_pct"].sum(), 2),
        "avg_win_pct": round(wins["return_pct"].mean(), 2) if len(wins) > 0 else None,
        "avg_loss_pct": round(losses["return_pct"].mean(), 2) if len(losses) > 0 else None,
    }


def get_recent_signals(limit: int = 50) -> pd.DataFrame:
    init_db()
    conn = _get_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM signals ORDER BY timestamp DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df