"""
Telegram alerting – free via Bot API.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import requests

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_message(text: str, chat_id: Optional[str] = None, parse_mode: str = "HTML") -> bool:
    """
    Send a plain message to the configured Telegram chat.
    """
    token = TELEGRAM_BOT_TOKEN
    chat = chat_id or TELEGRAM_CHAT_ID

    if not token or not chat:
        logger.warning("Telegram token or chat_id not set – skipping send")
        print(f"[TELEGRAM MOCK] {text}")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        logger.error(f"Telegram API error: {resp.status_code} – {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def format_signal(sig: Dict) -> str:
    """Pretty HTML message for one signal."""
    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig["signal"], "⚪")
    conf = sig.get("confidence", 0)
    conf_pct = f"{conf:.0%}" if conf <= 1 else f"{conf:.1f}"
    horizon = sig.get("horizon_days") or 5
    pos = sig.get("position_size_pct")
    pos_str = f"{pos:.1%}" if pos else "–"

    lines = [
        f"{emoji} <b>{sig['signal']}</b> – <code>{sig['ticker']}</code>",
        f"Price: <b>${sig.get('price', 'N/A')}</b>",
        f"Confidence: <b>{conf_pct}</b>",
        f"Horizon: <b>{horizon} trading days</b>",
        f"Suggested size: <b>{pos_str}</b> of portfolio",
        f"Reason: {sig.get('reason', '')}",
        f"<i>{sig.get('timestamp', '')[:19]} UTC</i>",
    ]
    return "\n".join(lines)


def send_signals(signals: List[Dict], header: str = "📊 New Stock Signals") -> int:
    """
    Send a batch of signals. Returns number of messages successfully sent.
    """
    if not signals:
        send_message("No actionable signals at this time.")
        return 0

    # Header
    send_message(f"<b>{header}</b>\n{len(signals)} signal(s) found")

    sent = 0
    for sig in signals:
        msg = format_signal(sig)
        if send_message(msg):
            sent += 1
    return sent


def send_daily_summary(signals: List[Dict], metrics: Optional[Dict] = None) -> bool:
    """Optional richer daily summary."""
    buys = [s for s in signals if s["signal"] == "BUY"]
    sells = [s for s in signals if s["signal"] == "SELL"]

    text = (
        f"<b>📅 Daily Signal Summary</b>\n\n"
        f"🟢 Buys: {len(buys)}\n"
        f"🔴 Sells: {len(sells)}\n"
    )
    if metrics:
        text += f"\nModel accuracy (last train): {metrics.get('accuracy', 0):.1%}"
    return send_message(text)
