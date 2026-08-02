"""
Mobile-friendly Streamlit dashboard for Stock Signals.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from config.settings import DEFAULT_WATCHLIST, DASHBOARD_TITLE, MIN_CONFIDENCE, SIGNAL_HORIZON_DAYS
from data.fetcher import fetch_ohlcv
from signals.generator import generate_signals, generate_signal_for_ticker, train_on_watchlist
from signals.tracker import get_performance_stats, get_recent_signals, evaluate_open_signals, init_db
from models.trainer import load_model
from alerts.telegram_bot import send_signals

st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { background: #1e1e1e; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Signals")
st.caption(f"Horizon: {SIGNAL_HORIZON_DAYS} trading days · Layer 1 filters · Tracking enabled")

# Sidebar
with st.sidebar:
    st.header("Controls")
    watchlist = st.multiselect(
        "Watchlist",
        options=DEFAULT_WATCHLIST,
        default=DEFAULT_WATCHLIST[:12],
    )
    min_conf = st.slider("Min confidence", 0.50, 0.75, MIN_CONFIDENCE, 0.01)
    only_actionable = st.checkbox("Only BUY / SELL", value=True)

    st.divider()
    if st.button("🔄 Generate Signals", type="primary", use_container_width=True):
        st.session_state["run_signals"] = True

    if st.button("🤖 Retrain Model", use_container_width=True):
        with st.spinner("Training..."):
            try:
                metrics = train_on_watchlist(watchlist or DEFAULT_WATCHLIST)
                st.success(f"Trained · Acc {metrics['accuracy']:.1%}")
            except Exception as e:
                st.error(str(e))

    if st.button("📤 Send to Telegram", use_container_width=True):
        if "last_signals" in st.session_state and st.session_state["last_signals"]:
            n = send_signals(st.session_state["last_signals"])
            st.success(f"Sent {n} signals")
        else:
            st.warning("Generate signals first")

    if st.button("📊 Evaluate Open Signals", use_container_width=True):
        n = evaluate_open_signals()
        st.success(f"Evaluated {n} signals")

    st.divider()
    model_data = load_model()
    if model_data:
        m = model_data.get("metrics", {})
        st.caption(f"Model Acc: {m.get('accuracy', 0):.1%}")
    else:
        st.warning("No model found")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Signals", "📈 Performance", "📊 Chart", "ℹ️ About"])

with tab1:
    if st.session_state.get("run_signals") or "last_signals" not in st.session_state:
        with st.spinner("Generating signals..."):
            try:
                signals = generate_signals(
                    tickers=watchlist or DEFAULT_WATCHLIST,
                    min_confidence=min_conf,
                    only_actionable=only_actionable,
                )
                for s in signals:
                    s["horizon_days"] = SIGNAL_HORIZON_DAYS
                st.session_state["last_signals"] = signals
                st.session_state["run_signals"] = False
            except Exception as e:
                st.error(str(e))
                signals = []
    else:
        signals = st.session_state.get("last_signals", [])

    if not signals:
        st.info("No actionable signals right now.")
    else:
        buys = sum(1 for s in signals if s["signal"] == "BUY")
        sells = sum(1 for s in signals if s["signal"] == "SELL")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signals", len(signals))
        c2.metric("Buys", buys)
        c3.metric("Sells", sells)
        c4.metric("Horizon", f"{SIGNAL_HORIZON_DAYS}d")

        st.divider()
        for sig in signals:
            color = "green" if sig["signal"] == "BUY" else "red"
            col1, col2, col3 = st.columns([2, 2, 3])
            with col1:
                st.markdown(f"### :{color}[{sig['signal']}] `{sig['ticker']}`")
            with col2:
                st.metric("Price", f"${sig['price']}" if sig.get("price") else "–")
                st.caption(f"Conf: {sig['confidence']:.0%}")
            with col3:
                st.write(sig.get("reason", ""))
                if sig.get("position_size_pct"):
                    st.caption(f"Suggested size: {sig['position_size_pct']:.1%}")
            st.divider()

with tab2:
    st.subheader("Success / Failure Tracking")
    init_db()
    stats = get_performance_stats()

    if stats["n_closed"] == 0:
        st.info("No closed signals yet. They are evaluated after 5 trading days.")
        st.write("Click **Evaluate Open Signals** in the sidebar after a few days.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Closed", stats["n_closed"])
        c2.metric("Win rate", f"{stats['win_rate']}%" if stats["win_rate"] is not None else "–")
        c3.metric("Avg return", f"{stats['avg_return_pct']}%" if stats["avg_return_pct"] is not None else "–")
        c4.metric("Total return", f"{stats['total_return_pct']}%" if stats["total_return_pct"] is not None else "–")

    st.divider()
    st.subheader("Recent signals log")
    recent = get_recent_signals(40)
    if not recent.empty:
        st.dataframe(
            recent[["timestamp", "ticker", "signal", "confidence", "entry_price", "horizon_days", "status", "return_pct", "success"]],
            use_container_width=True
        )
    else:
        st.caption("No signals logged yet.")

with tab3:
    ticker = st.selectbox("Select ticker", options=watchlist or DEFAULT_WATCHLIST)
    if ticker:
        df = fetch_ohlcv(ticker, period="6mo")
        if not df.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name=ticker
            )])
            fig.update_layout(title=f"{ticker}", xaxis_rangeslider_visible=False, height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown(f"""
    ### Current Setup
    - **Horizon**: {SIGNAL_HORIZON_DAYS} trading days
    - **Filters**: SPY regime + Volume + Trend + Top 5 signals
    - **Tracking**: Every signal is saved and later evaluated
    - **Watchlist**: ~45 liquid stocks + ETFs
    """)

st.caption(f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")