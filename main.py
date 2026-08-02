"""
Main entry point – generate signals, log them, and optionally send to Telegram.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import DEFAULT_WATCHLIST, LOGS_DIR, SIGNAL_HORIZON_DAYS
from signals.generator import generate_signals, train_on_watchlist
from signals.tracker import log_signals, evaluate_open_signals, get_performance_stats, init_db
from alerts.telegram_bot import send_signals, send_daily_summary
from models.trainer import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "signals.log"),
    ],
)
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser(description="Stock Signals Engine")
    parser.add_argument("--train", action="store_true", help="Retrain model on watchlist")
    parser.add_argument("--send", action="store_true", help="Send signals to Telegram")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate open signals that reached horizon")
    parser.add_argument("--stats", action="store_true", help="Show performance statistics")
    parser.add_argument("--min-conf", type=float, default=0.58, help="Minimum confidence")
    parser.add_argument("--tickers", nargs="+", help="Override watchlist")
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_WATCHLIST
    init_db()

    if args.train:
        logger.info("Training model...")
        metrics = train_on_watchlist(tickers)
        logger.info(f"Training done: {metrics}")
        return

    if args.evaluate:
        n = evaluate_open_signals()
        logger.info(f"Evaluated {n} signals")
        stats = get_performance_stats()
        logger.info(f"Performance: {stats}")
        return

    if args.stats:
        stats = get_performance_stats()
        print("\n=== Performance Statistics ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if load_model() is None:
        logger.info("No model found – training first...")
        train_on_watchlist(tickers)

    logger.info(f"Generating signals for {len(tickers)} tickers (horizon={SIGNAL_HORIZON_DAYS} days)...")
    signals = generate_signals(tickers=tickers, min_confidence=args.min_conf, only_actionable=True)

    for s in signals:
        s["horizon_days"] = SIGNAL_HORIZON_DAYS
        s["horizon_label"] = f"{SIGNAL_HORIZON_DAYS} trading days"

    if not signals:
        logger.info("No actionable signals")
        if args.send:
            send_signals([])
        return

    n_logged = log_signals(signals, horizon_days=SIGNAL_HORIZON_DAYS)
    logger.info(f"Logged {n_logged} signals for future tracking")

    logger.info(f"Found {len(signals)} signals:")
    for s in signals:
        logger.info(f"  {s['signal']:4} {s['ticker']:5} conf={s['confidence']:.2f} price={s['price']} | horizon={SIGNAL_HORIZON_DAYS}d")

    if args.send:
        n = send_signals(signals)
        logger.info(f"Telegram: {n} messages sent")
        send_daily_summary(signals)


if __name__ == "__main__":
    main()