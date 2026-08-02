"""
Central configuration for the Stock Signals app.
All free / open options.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "data" / "signals.db"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Expanded watchlist (~45 liquid US stocks + major ETFs)
DEFAULT_WATCHLIST = [
    # Mega tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CRM",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
    # Finance
    "JPM", "V", "MA", "BAC", "WFC", "GS",
    # Healthcare / Consumer
    "JNJ", "UNH", "LLY", "ABBV", "WMT", "COST", "PG", "KO", "PEP", "HD", "MCD",
    # Energy / Industrial
    "XOM", "CVX", "BA", "CAT", "GE",
    # Semiconductors & others
    "AMD", "INTC", "QCOM", "TXN", "NFLX", "DIS", "CMCSA", "T", "VZ",
    # Extra liquid names
    "ADBE", "NKE", "SBUX", "LOW", "UPS"
]

# Signal settings
LOOKBACK_DAYS = 365 * 2
MIN_CONFIDENCE = 0.58
SIGNAL_COOLDOWN_HOURS = 24
SIGNAL_HORIZON_DAYS = 5

# ----- Layer 1 filters -----
MAX_SIGNALS_PER_RUN = 5          # only keep the strongest N signals
REQUIRE_SPY_TREND = True         # block BUY in bear / SELL in bull
REQUIRE_VOLUME_CONFIRM = True    # require volume >= average
REQUIRE_TREND_FILTER = True      # basic trend alignment

# Risk defaults (paper)
MAX_POSITION_PCT = 0.05
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# Model
MODEL_NAME = "lightgbm_signal_v1"
RETRAIN_EVERY_DAYS = 7

# Dashboard
DASHBOARD_TITLE = "Stock Signals Dashboard"