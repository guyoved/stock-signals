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
import streamlit as st

try:
    # First try Streamlit Cloud secrets
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
except Exception:
    # Fallback to .env (for local use)
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Expanded watchlist (~45 liquid US stocks + major ETFs)
DEFAULT_WATCHLIST = [
    # ===== Mega / Big Tech =====
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "CRM", "ADBE", "NFLX", "AMD", "INTC", "QCOM", "TXN", "CSCO", "IBM", "NOW",
    "INTU", "AMAT", "LRCX", "KLAC", "MU", "SNPS", "CDNS", "ANET", "PANW", "CRWD",

    # ===== AI / Semiconductor / Growth (Balanced) =====
    "CRDO", "ARM", "SMCI", "DELL", "HPE", "NTAP", "WDC", "STX", "ON", "MPWR",
    "MRVL", "SWKS", "QRVO", "LSCC", "RMBS", "SITM", "ALGM", "DIOD", "FORM", "ACLS",
    "PLTR", "SNOW", "DDOG", "NET", "CLOUD", "ZS", "OKTA", "ESTC", "MDB", "PATH",
    "AI", "SOUN", "BBAI", "UPST", "AFRM", "HOOD", "COIN", "MSTR", "RIOT", "MARA",

    # ===== ETFs =====
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    "XLU", "XLB", "XLRE", "SMH", "SOXX", "BOTZ", "ARKK", "ARKW", "ARKF", "VGT",

    # ===== Finance =====
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
    "SCHW", "USB", "PNC", "TFC", "COF", "AIG", "MET", "PRU", "TRV", "ALL",

    # ===== Healthcare / Biotech =====
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "REGN", "VRTX", "ISRG", "SYK", "BSX", "MDT", "ZTS", "CI",

    # ===== Consumer / Retail =====
    "WMT", "COST", "PG", "KO", "PEP", "HD", "MCD", "NKE", "SBUX", "LOW",
    "TGT", "DIS", "CMCSA", "T", "VZ", "PM", "MO", "CL", "EL", "MDLZ",
    "BKNG", "MAR", "HLT", "YUM", "CMG", "DPZ", "ROST", "TJX", "BBY", "ULTA",

    # ===== Energy / Industrial =====
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "HAL",
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "UNP", "CSX",
    "NSC", "FDX", "WM", "RSG", "ETN", "EMR", "ITW", "PH", "ROK", "DOV",

    # ===== Other strong / liquid names =====
    "BRK-B", "ACN", "SPGI", "MCO", "ICE", "CME", "MSCI", "NDAQ", "FICO", "ADP",
    "PAYX", "CTSH", "FIS", "FISV", "GPN", "SQ", "PYPL", "SHOP", "MELI", "SE",
    "JD", "PDD", "BABA", "NIO", "XPEV", "LI", "BIDU", "TSM", "ASML", "SAP"
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