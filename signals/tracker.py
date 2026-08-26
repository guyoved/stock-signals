"""
Permanent signal tracking using Supabase.
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _get_supabase():
    """Create Supabase client from environment / Streamlit secrets."""
    url = ""
    key = ""

    # Try Streamlit secrets first
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        pass

    # Fallback to environment variables (GitHub Actions + local)
    if not url:
        url = os.getenv("SUPABASE_URL", "")
    if not key:
        key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is missing")

    from supabase import create_client
    return create_client(url, key)


def init_db() -> None:
    """No-op for Supabase (table already exists)."""
    logger.info("Using Supabase for permanent tracking")


def log_signal(sig: Dict[str, Any], horizon_days: int = 5) -> None:
    """Insert a new signal into Supabase."""
    try:
        sb = _get_supabase()
        row = {
            "timestamp": sig.get("timestamp") or datetime.utcnow().isoformat(),
            "ticker": sig["ticker"],
            "signal": sig["signal"],
            "confidence": sig.get("confidence"),
            "entry_price": sig.get("price"),
            "horizon_days": horizon_days,
            "reason": sig.get("reason"),
            "status": "open",
        }
        sb.table("signals").insert(row).execute()
        logger.info(f"Logged signal: {sig['ticker']} {sig['signal']}")
    except Exception as e:
        logger.error(f"Failed to log signal: {e}")


def log_signals(signals: List[Dict[str, Any]], horizon_days: int = 5) -> int:
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
        return float(df["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch exit price for {ticker}: {e}")
        return None


def evaluate_open_signals(min_days: int = 4) -> int:
    """Evaluate open signals that are old enough."""
    try:
        sb = _get_supabase()
        result = sb.table("signals").select("*").eq("status", "open").execute()
        rows = result.data or []
        logger.info(f"Found {len(rows)} open signals")

        evaluated = 0
        now = datetime.utcnow()

        for row in rows:
            try:
                entry_time = datetime.fromisoformat(row["timestamp"][:19])
                days_passed = (now - entry_time).days
                logger.info(f"{row['ticker']}: {days_passed} days passed")

                if days_passed < min_days:
                    continue

                exit_price = _get_price_on_or_after(row["ticker"], row["timestamp"])
                if exit_price is None or row.get("entry_price") is None:
                    logger.warning(f"No exit price for {row['ticker']}")
                    continue

                entry = float(row["entry_price"])
                if row["signal"] == "BUY":
                    ret = (exit_price / entry) - 1
                else:
                    ret = (entry / exit_price) - 1

                success = 1 if ret > 0 else 0

                sb.table("signals").update({
                    "status": "closed",
                    "exit_price": exit_price,
                    "exit_date": now.isoformat(),
                    "return_pct": round(ret * 100, 3),
                    "success": success,
                    "evaluated_at": now.isoformat(),
                }).eq("id", row["id"]).execute()

                evaluated += 1
                logger.info(f"Evaluated {row['ticker']}: {ret*100:.2f}% success={success}")

            except Exception as e:
                logger.error(f"Error evaluating {row.get('ticker')}: {e}")

        return evaluated

    except Exception as e:
        logger.error(f"evaluate_open_signals failed: {e}")
        return 0


def get_performance_stats() -> Dict[str, Any]:
    try:
        sb = _get_supabase()
        result = sb.table("signals").select("*").eq("status", "closed").execute()
        rows = result.data or []

        if not rows:
            return {
                "n_closed": 0, "win_rate": None, "avg_return_pct": None,
                "total_return_pct": None, "n_wins": 0, "n_losses": 0,
            }

        df = pd.DataFrame(rows)
        wins = df[df["success"] == 1]
        losses = df[df["success"] == 0]

        return {
            "n_closed": len(df),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_rate": round(len(wins) / len(df) * 100, 1),
            "avg_return_pct": round(df["return_pct"].mean(), 2),
            "total_return_pct": round(df["return_pct"].sum(), 2),
            "avg_win_pct": round(wins["return_pct"].mean(), 2) if len(wins) > 0 else None,
            "avg_loss_pct": round(losses["return_pct"].mean(), 2) if len(losses) > 0 else None,
        }
    except Exception as e:
        logger.error(f"get_performance_stats failed: {e}")
        return {"n_closed": 0, "win_rate": None, "avg_return_pct": None,
                "total_return_pct": None, "n_wins": 0, "n_losses": 0}


def get_recent_signals(limit: int = 50) -> pd.DataFrame:
    try:
        sb = _get_supabase()
        result = (
            sb.table("signals")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        rows = result.data or []
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        logger.error(f"get_recent_signals failed: {e}")
        return pd.DataFrame()