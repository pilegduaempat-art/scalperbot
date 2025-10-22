# app.py
"""
Streamlit dashboard + Telegram notifier for automated analysis of top volatile Binance Futures pairs.

Features:
- Fetch top N volatile pairs (by 24h price change or ATR)
- Calculate ATR, approximate CVD (from aggTrades), fetch Open Interest & Funding Rate
- Generate recommendation signals (Scalp Long / Scalp Short / Wait)
- Streamlit UI with charts and auto-refresh
- Telegram notifications for new signals
- Custom API settings in dashboard

Requirements:
pip install streamlit pandas requests plotly python-dotenv
"""

import os
import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime, timedelta

# -------------------------
# Configuration
# -------------------------
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# App defaults
DEFAULT_TOP_N = 10
DEFAULT_TIMEFRAME = "15m"
REFRESH_SECONDS = 60
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

# Fallback symbols if API fails
FALLBACK_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT",
    "LINKUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "MATICUSDT"
]

# Load from Streamlit secrets if available, otherwise use empty defaults
try:
    DEFAULT_COINGLASS_KEY = st.secrets.get("COINGLASS_API_KEY", "")
    DEFAULT_TG_TOKEN = st.secrets.get("TG_BOT_TOKEN", "")
    DEFAULT_TG_CHAT_ID = st.secrets.get("TG_CHAT_ID", "")
except:
    DEFAULT_COINGLASS_KEY = os.getenv("COINGLASS_API_KEY", "")
    DEFAULT_TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
    DEFAULT_TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")


# -------------------------
# Binance API Functions
# -------------------------
def fetch_exchange_info():
    """Fetch exchange information from Binance Futures"""
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/exchangeInfo", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_24hr_ticker():
    """Fetch 24-hour ticker data for all symbols"""
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/ticker/24hr", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 500):
    """Fetch candlestick/kline data for a symbol"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/klines", params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df


def fetch_funding_rate(symbol: str, limit: int = 1):
    """Fetch funding rate for a symbol"""
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/fundingRate", params=params, timeout=10)
    r.raise_for_status()
    arr = r.json()
    return float(arr[-1]["fundingRate"]) if arr else 0.0


def fetch_open_interest(symbol: str):
    """Fetch open interest for a symbol"""
    params = {"symbol": symbol}
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/openInterest", params=params, timeout=10)
    r.raise_for_status()
    return float(r.json().get("openInterest", 0.0))


def fetch_agg_trades(symbol: str, start_time_ms: int = None, limit: int = 1000):
    """Fetch aggregated trades for CVD calculation"""
    params = {"symbol": symbol, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms
    r = requests.get(BINANCE_FUTURES_BASE.replace("fapi", "api") + "/v3/aggTrades", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# -------------------------
# Coinglass API (Optional)
# -------------------------
def fetch_coinglass_metric(symbol: str, metric: str, api_key: str = None):
    """Fetch metrics from Coinglass API (optional, placeholder)"""
    if not api_key:
        return None
    
    headers = {"coinglassSecret": api_key}
    
    try:
        url = f"https://openapi.coinglass.com/public/v2/futures/{metric}"
        r = requests.get(url, headers=headers, params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# -------------------------
# Technical Indicators
# -------------------------
def atr(df: pd.DataFrame, period: int = 14):
    """Calculate Average True Range (ATR)"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(period, min_periods=1).mean()
    return atr_series


def approximate_cvd_from_aggtrades(symbol: str, minutes: int = 60):
    """Approximate cumulative volume delta"""
    try:
        df = fetch_klines(symbol, interval="1m", limit=minutes)
        df["taker_buy_base"] = df["taker_buy_base"].astype(float)
        df["delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
        cvd = df["delta"].cumsum().iloc[-1]
        return cvd
    except Exception:
        return None


# -------------------------
# Signal Generation
# -------------------------
def generate_recommendation(symbol: str, tf: str = "15m"):
    """Generate trading recommendation based on multiple factors"""
    try:
        kl = fetch_klines(symbol, interval=tf, limit=50)
        last = kl.iloc[-1]
        recent_atr = atr(kl, period=14).iloc[-1]
        funding = fetch_funding_rate(symbol)
        oi = fetch_open_interest(symbol)
        cvd = approximate_cvd_from_aggtrades(symbol, minutes=60)

        price = float(last["close"])
        atr_pct = recent_atr / price if price else 0.0
        big_sell = (cvd is not None and cvd < 0 and abs(cvd) > 0)

        signal = "WAIT"
        reason = []
        entry = None
        tp = None
        sl = None
        rr = None

        # Scalp Long logic
        if funding < -0.0005 and (cvd is not None and cvd < 0):
            signal = "SCALP LONG"
            entry = price
            tp = price * (1 + max(0.01, min(0.05, atr_pct * 5)))
            sl = price * (1 - max(0.01, min(0.03, atr_pct * 3)))
            reason.append("Funding negative (shorts dominate) + selling pressure -> potential short squeeze")
        # Scalp Short logic
        elif funding > 0.0005 and (cvd is not None and cvd > 0):
            signal = "SCALP SHORT"
            entry = price
            tp = price * (1 - max(0.01, min(0.05, atr_pct * 5)))
            sl = price * (1 + max(0.01, min(0.03, atr_pct * 3)))
            reason.append("Funding positive + buying pressure -> possible extension to downside reversal")
        else:
            if atr_pct > 0.03:
                signal = "WAIT"
                reason.append("High volatility – prefer to wait for clearer setup")
            else:
                signal = "WAIT"
                reason.append("No clear short-squeeze or reversal condition")

        if entry and tp and sl:
            potential = (tp - entry) / entry if "LONG" in signal else (entry - tp) / entry
            loss = abs((sl - entry) / entry)
            rr = round((potential / loss) if loss != 0 else None, 2)

        return {
            "symbol": symbol,
            "price": price,
            "funding": funding,
            "oi": oi,
            "cvd": cvd,
            "atr": recent_atr,
            "atr_pct": atr_pct,
            "signal": signal,
            "reason": " ; ".join(reason),
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "rr": rr,
            "ts": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def get_top_n_pairs_by_volatility(n=10, tf="15m"):
    """Get top N most volatile trading pairs"""
    try:
        tickers = fetch_24hr_ticker()
        fut = [t for t in tickers if t["symbol"].endswith("USDT") and ("PERP" not in t)]
        df = pd.DataFrame(fut)
        df["priceChangePercent"] = df["priceChangePercent"].astype(float)
        df["abs_change"] = df["priceChangePercent"].abs()
        top = df.sort_values("abs_change", ascending=False).head(n)
        return top["symbol"].tolist()
    except Exception:
        return FALLBACK_SYMBOLS[:n]


# -------------------------
# Telegram Functions
# -------------------------
def send_telegram_message(text: str, bot_token: str = None, chat_id: str = None):
    """Send message via Telegram bot"""
    if not bot_token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Binance Futures Auto-Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Binance Futures Auto-Analysis – Top Volatile Pairs + Telegram Alerts")

# -------------------------
# Sidebar: Settings & API Configuration
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Analysis Settings
    st.subheader("Analysis Parameters")
    top_n = st.number_input("Top N pairs", min_value=1, max_value=50, value=DEFAULT_TOP_N)
    timeframe = st.selectbox("Indicator timeframe (for ATR / signals)", TIMEFRAMES, index=TIMEFRAMES.index(DEFAULT_TIMEFRAME))
    refresh = st.number_input("Auto-refresh (seconds)", min_value=10, max_value=600, value=REFRESH_SECONDS)
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    # Coinglass API
    with st.expander("Coinglass API (Optional)", expanded=False):
        st.info("Coinglass API provides additional market data. Leave empty if not using.")
        coinglass_key = st.text_input(
            "Coinglass API Key",
            value=DEFAULT_COINGLASS_KEY,
            type="password",
            help="Get your API key from https://www.coinglass.com"
        )
        coinglass_status = "✓ configured" if coinglass_key else "✗ not set"
        st.caption(f"Status: {coinglass_status}")
    
    # Telegram Configuration
    with st.expander("Telegram Notifications", expanded=False):
        st.info("Configure Telegram to receive trading signal alerts.")
        tg_bot_token = st.text_input(
            "Telegram Bot Token",
            value=DEFAULT_TG_TOKEN,
            type="password",
            help="Get bot token from @BotFather on Telegram"
        )
        tg_chat_id = st.text_input(
            "Telegram Chat ID",
            value=DEFAULT_TG_CHAT_ID,
            help="Your personal chat ID or group chat ID"
        )
        notify = st.checkbox(
            "Enable Telegram notifications",
            value=bool(tg_bot_token and tg_chat_id)
        )
        
        tg_status = "✓ configured" if (tg_bot_token and tg_chat_id) else "✗ not set"
        st.caption(f"Status: {tg_status}")
        
        if st.button("🧪 Test Telegram Connection"):
            if tg_bot_token and tg_chat_id:
                test_msg = "🔔 Test message from Binance Futures Auto-Analysis Dashboard"
                success = send_telegram_message(test_msg, tg_bot_token, tg_chat_id)
                if success:
                    st.success("✅ Telegram connection successful!")
                else:
                    st.error("❌ Failed to send message. Check your credentials.")
            else:
                st.warning("⚠️ Please enter both Bot Token and Chat ID first.")
    
    st.markdown("---")
    
    # Manual Refresh Button
    manual_refresh = st.button("🔄 Refresh Now", use_container_width=True)
    
    st.markdown("---")
    
    # Environment Status
    st.subheader("📊 System Status")
    st.caption(f"Coinglass API: {coinglass_status}")
    st.caption(f"Telegram: {tg_status}")
    st.caption(f"Auto-refresh: Every {refresh}s")

# -------------------------
# Main Content Area
# -------------------------
col1, col2 = st.columns([3, 1])

with col2:
    st.metric("Top Pairs", top_n)
    st.metric("Timeframe", timeframe)
    st.metric("Refresh", f"{refresh}s")

with col1:
    placeholder = st.empty()

# -------------------------
# Session State Management
# -------------------------
if "last_signals" not in st.session_state:
    st.session_state["last_signals"] = {}

if "last_update" not in st.session_state:
    st.session_state["last_update"] = 0

# -------------------------
# Main Analysis Loop
# -------------------------
def main_loop():
    """Main analysis and display loop"""
    current_time = time.time()
    
    # Check if it's time to refresh
    if current_time - st.session_state["last_update"] >= refresh or manual_refresh:
        st.session_state["last_update"] = current_time
        
        try:
            # Fetch top volatile pairs
            symbols = get_top_n_pairs_by_volatility(n=top_n, tf=timeframe)
            results = []
            
            # Generate recommendations for each symbol
            with st.spinner(f"Analyzing {len(symbols)} pairs..."):
                for s in symbols:
                    res = generate_recommendation(s, tf=timeframe)
                    results.append(res)

            df = pd.DataFrame(results)
            
            # Display results
            with placeholder.container():
                st.markdown(f"### 📈 Top {len(df)} volatile pairs analysis")
                st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
                
                # Prepare display dataframe - FIXED: Check if columns exist before selecting
                available_cols = []
                expected_cols = ["symbol", "price", "atr", "atr_pct", "funding", "oi", "cvd", "signal", "rr", "reason", "ts"]
                
                for col in expected_cols:
                    if col in df.columns:
                        available_cols.append(col)
                
                # Create display dataframe with available columns only
                if available_cols:
                    display_df = df[available_cols].copy()
                    
                    # Format numeric columns if they exist
                    if "price" in display_df.columns:
                        display_df["price"] = display_df["price"].map(lambda x: round(x, 6) if pd.notnull(x) else x)
                    if "atr" in display_df.columns:
                        display_df["atr"] = display_df["atr"].map(lambda x: round(x, 6) if pd.notnull(x) else x)
                    if "atr_pct" in display_df.columns:
                        display_df["atr_pct"] = display_df["atr_pct"].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else x)
                    if "funding" in display_df.columns:
                        display_df["funding"] = display_df["funding"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else x)
                    if "oi" in display_df.columns:
                        display_df["oi"] = display_df["oi"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
                    
                    # Color-code signals
                    def highlight_signal(row):
                        if 'signal' in row and row['signal'] == 'SCALP LONG':
                            return ['background-color: #d4edda'] * len(row)
                        elif 'signal' in row and row['signal'] == 'SCALP SHORT':
                            return ['background-color: #f8d7da'] * len(row)
                        else:
                            return [''] * len(row)
                    
                    st.dataframe(
                        display_df.style.apply(highlight_signal, axis=1),
                        height=420,
                        use_container_width=True
                    )
                else:
                    st.warning("No data available to display")
                
                # Per-symbol expanders with charts
                st.markdown("---")
                st.subheader("📊 Detailed Analysis")
                
                for r in results:
                    if not isinstance(r, dict):
                        continue
                        
                    sym = r.get("symbol")
                    sig = r.get("signal")
                    price = r.get("price")
                    
                    # Color indicator for signal type
                    if sig == "SCALP LONG":
                        indicator = "🟢"
                    elif sig == "SCALP SHORT":
                        indicator = "🔴"
                    else:
                        indicator = "⚪"
                    
                    with st.expander(f"{indicator} {sym} – {sig} – ${price}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**📊 Signal Information**")
                            st.write(f"**Signal:** {r.get('signal')}")
                            st.write(f"**Price:** ${r.get('price')}")
                            st.write(f"**Funding Rate:** {r.get('funding')}")
                            st.write(f"**Open Interest:** {r.get('oi')}")
                            
                        with col_b:
                            st.write("**📈 Technical Details**")
                            st.write(f"**CVD (approx):** {r.get('cvd')}")
                            st.write(f"**ATR:** {r.get('atr')} ({r.get('atr_pct')})")
                            if r.get("entry"):
                                st.write(f"**Entry:** ${r.get('entry'):.6f}")
                                st.write(f"**TP:** ${r.get('tp'):.6f}")
                                st.write(f"**SL:** ${r.get('sl'):.6f}")
                                st.write(f"**Risk/Reward:** {r.get('rr')}")
                        
                        st.write("**💡 Reason:**", r.get("reason"))
                        
                        # Candlestick chart
                        try:
                            kl = fetch_klines(sym, interval=timeframe, limit=200)
                            fig = go.Figure(data=[go.Candlestick(
                                x=kl["open_time"],
                                open=kl["open"],
                                high=kl["high"],
                                low=kl["low"],
                                close=kl["close"],
                                name="Price"
                            )])
                            fig.update_layout(
                                height=300,
                                margin=dict(l=0, r=0, t=20, b=0),
                                xaxis_title="Time",
                                yaxis_title="Price",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Chart error: {e}")

            # Telegram Notifications
            if notify and tg_bot_token and tg_chat_id:
                for r in results:
                    if not isinstance(r, dict):
                        continue
                        
                    s = r.get("symbol")
                    sig = r.get("signal")
                    last = st.session_state["last_signals"].get(s)
                    
                    # Send notification for new actionable signals
                    if sig in ["SCALP LONG", "SCALP SHORT"]:
                        if last != sig:
                            st.session_state["last_signals"][s] = sig
                            text = (
                                f"🔔 *New Signal Alert*\n\n"
                                f"*Signal:* {sig}\n"
                                f"*Pair:* {s}\n"
                                f"*Price:* ${r.get('price')}\n"
                                f"*Entry:* ${r.get('entry'):.6f}\n"
                                f"*TP:* ${r.get('tp'):.6f}\n"
                                f"*SL:* ${r.get('sl'):.6f}\n"
                                f"*RRR:* {r.get('rr')}\n"
                                f"*Reason:* {r.get('reason')}"
                            )
                            
                            success = send_telegram_message(text, tg_bot_token, tg_chat_id)
                            
                            if success:
                                st.toast(f"✅ Telegram alert sent for {s}", icon="✅")
                            else:
                                st.toast(f"⚠️ Failed to send Telegram alert for {s}", icon="⚠️")
                    else:
                        # Clear previous signal if resolved
                        if st.session_state["last_signals"].get(s) and sig == "WAIT":
                            st.session_state["last_signals"].pop(s, None)

        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.exception(e)

# Run main loop
main_loop()

# Auto-refresh
time.sleep(1)
st.rerun()

# Footer
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This is a starter dashboard for educational purposes. "
    "Tune thresholds, add additional indicators, and backtest rules before live trading. "
    "Always use testnet and small position sizes. Not financial advice."
)
