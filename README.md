# Stock Signals App

Free end-to-end stock buy/sell signal system with **Telegram alerts** and **mobile web dashboard**.

## Features
- Free market data via Yahoo Finance (`yfinance`)
- Technical indicators + LightGBM ML model
- BUY / SELL / HOLD signals with confidence scores
- Telegram notifications
- Mobile-friendly Streamlit dashboard
- Simple backtester
- Paper-trading ready

## Quick Start

### 1. Install dependencies
```bash
cd stock_signals
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Telegram (optional but recommended)
```bash
cp .env.example .env
# Edit .env with your bot token and chat_id
```

How to get them:
1. Message `@BotFather` on Telegram → `/newbot` → copy the token
2. Message `@userinfobot` → copy your chat id
3. Start a conversation with your new bot (press Start)

### 3. Train the model (first time)
```bash
python main.py --train
```

### 4. Generate signals
```bash
python main.py                # just print
python main.py --send         # also send to Telegram
```

### 5. Launch the mobile dashboard
```bash
streamlit run dashboard/app.py
```
Open the URL on your iPhone (same Wi-Fi or use a free tunnel later).

## Project Structure
```
stock_signals/
├── config/settings.py      # All configuration
├── data/fetcher.py         # yfinance data
├── features/engineering.py # Technical indicators
├── models/trainer.py       # LightGBM train & predict
├── signals/generator.py    # Signal logic
├── alerts/telegram_bot.py  # Telegram sender
├── backtest/               # Simple validation
├── dashboard/app.py        # Streamlit UI
├── main.py                 # CLI entry point
└── requirements.txt
```

## Important Notes
- This is **not financial advice**. Use at your own risk.
- Always start with paper trading / tiny size.
- Markets are noisy – no model is consistently accurate.
- Retrain periodically (`python main.py --train`).

## Next improvements (easy to add later)
- More tickers / crypto
- Intraday data
- Better walk-forward backtesting
- Position sizing & portfolio risk
- Scheduled runs (cron or GitHub Actions free tier)
