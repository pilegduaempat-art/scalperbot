# app.py
"""
Streamlit dashboard + Telegram notifier for automated analysis of top volatile Binance Futures pairs.
Features:
- Fetch top N volatile pairs (by 24h price change or ATR)
- Calculate ATR, approximate CVD (from aggTrades), fetch Open Interest & Funding Rate
- Generate recommendation signals (Scalp Long / Scalp Short / Wait)
- Streamlit UI with charts and auto-refresh
- Telegram notifications for new signals
Requirements:
pip install streamlit pandas requests plotly python-dotenv
Optional: install ccxt if you prefer (not required here)
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

# CONFIG / ENV
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")  # optional
TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")  # optional
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")  # optional

# App defaults
DEFAULT_TOP_N = 10
DEFAULT_TIMEFRAME = "15m"  # used for ATR/ohlcv
REFRESH_SECONDS = 60


# -------------------------
# Helper: Binance REST calls
# -------------------------
def fetch_exchange_info():
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/exchangeInfo", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_24hr_ticker():
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/ticker/24hr", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 500):
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
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/fundingRate", params=params, timeout=10)
    r.raise_for_status()
    arr = r.json()
    return float(arr[-1]["fundingRate"]) if arr else 0.0


def fetch_open_interest(symbol: str):
    params = {"symbol": symbol}
    r = requests.get(BINANCE_FUTURES_BASE + "/fapi/v1/openInterest", params=params, timeout=10)
    r.raise_for_status()
    return float(r.json().get("openInterest", 0.0))


def fetch_agg_trades(symbol: str, start_time_ms: int = None, limit: int = 1000):
    # approximate CVD by aggregated trades (isBuyerMaker flips)
    params = {"symbol": symbol, "limit": limit}
    if start_time_ms:
        params["startTime"] = start_time_ms
    r = requests.get(BINANCE_FUTURES_BASE.replace("fapi", "api") + "/v3/aggTrades", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# -------------------------
# Coinglass (optional)
# -------------------------
def fetch_coinglass_metric(symbol: str, metric: str):
    """
    Placeholder: fetch from Coinglass (if you have key & correct endpoint).
    Many Coinglass endpoints are paid; this function is optional.
    """
    if not COINGLASS_API_KEY:
        return None
    headers = {"coinglassSecret": COINGLASS_API_KEY}
    # Examples: endpoints must be replaced with real Coinglass endpoints if available.
    try:
        url = f"https://openapi.coinglass.com/public/v2/futures/{metric}"
        # This is illustrative. Check Coinglass docs to match endpoints and parameters.
        r = requests.get(url, headers=headers, params={"symbol": symbol})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# -------------------------
# Indicators & Signals
# -------------------------
def atr(df: pd.DataFrame, period: int = 14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(period, min_periods=1).mean()
    return atr_series


def approximate_cvd_from_aggtrades(symbol: str, minutes: int = 60):
    """
    Approximate cumulative volume delta using aggregated trades over 'minutes'.
    We use: if isBuyerMaker == True => trade was a sell (maker was buyer?), there's flip semantics on aggTrades:
    In aggTrades, there's no isBuyerMaker in the /aggTrades endpoint; instead trade data sometimes lacks taker info.
    We'll instead use a simple approximation: using taker buy metrics from futures klines (taker_buy_base)
    and compute delta over last N candles.
    """
    try:
        df = fetch_klines(symbol, interval="1m", limit=minutes)
        # Use taker_buy_base as proxy of aggressive buy volume; delta = taker_buy_base - (volume - taker_buy_base)
        df["taker_buy_base"] = df["taker_buy_base"].astype(float)
        df["delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
        cvd = df["delta"].cumsum().iloc[-1]
        return cvd
    except Exception:
        return None


def generate_recommendation(symbol: str, tf: str = "15m"):
    """
    Simple rule-based recommendation:
    - Use recent ATR (volatility)
    - Funding rate sign: negative funding suggests more shorts (potential short squeeze long)
    - OI trend: rising OI while price drops -> more shorts
    - CVD: negative big -> selling pressure
    Return dict: {signal, reason, entry, tp, sl, rr}
    """
    try:
        kl = fetch_klines(symbol, interval=tf, limit=50)
        last = kl.iloc[-1]
        recent_atr = atr(kl, period=14).iloc[-1]
        funding = fetch_funding_rate(symbol)
        oi = fetch_open_interest(symbol)
        cvd = approximate_cvd_from_aggtrades(symbol, minutes=60)  # 1h approx

        price = float(last["close"])
        # Basic thresholds (tuneable)
        atr_pct = recent_atr / price if price else 0.0
        big_sell = (cvd is not None and cvd < 0 and abs(cvd) > 0)  # placeholder condition

        signal = "WAIT"
        reason = []
        entry = None
        tp = None
        sl = None
        rr = None

        # Scalp Long logic:
        if funding < -0.0005 and (cvd is not None and cvd < 0):
            # Funding negative and selling pressure -> short-heavy market -> possible short squeeze
            signal = "SCALP LONG"
            entry = price
            tp = price * (1 + max(0.01, min(0.05, atr_pct * 5)))  # 1-5% target scaled by ATR
            sl = price * (1 - max(0.01, min(0.03, atr_pct * 3)))
            reason.append("Funding negative (shorts dominate) + selling pressure -> potential short squeeze")
        # Scalp Short logic:
        elif funding > 0.0005 and (cvd is not None and cvd > 0):
            signal = "SCALP SHORT"
            entry = price
            tp = price * (1 - max(0.01, min(0.05, atr_pct * 5)))
            sl = price * (1 + max(0.01, min(0.03, atr_pct * 3)))
            reason.append("Funding positive + buying pressure -> possible extension to downside reversal")
        else:
            # If large ATR and price dropped strongly -> consider WAIT or SHORT depending
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


# -------------------------
# Utilities
# -------------------------
def get_top_n_pairs_by_volatility(n=10, tf="15m"):
    # Use 24h tickers: fallback choose top movers by priceChangePercent magnitude
    try:
        tickers = fetch_24hr_ticker()
        # Filter perpetual futures symbols (USDT perpetual)
        fut = [t for t in tickers if t["symbol"].endswith("USDT") and ("PERP" not in t)]  # filtering heuristic
        df = pd.DataFrame(fut)
        df["priceChangePercent"] = df["priceChangePercent"].astype(float)
        df["abs_change"] = df["priceChangePercent"].abs()
        top = df.sort_values("abs_change", ascending=False).head(n)
        return top["symbol"].tolist()
    except Exception:
        # fallback list
        return ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "MATICUSDT"][:n]


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Binance Futures Auto-Analysis", layout="wide")
st.title("📊 Binance Futures Auto-Analysis – Top Volatile Pairs + Telegram Alerts")

col1, col2 = st.columns([3, 1])

with col2:
    st.write("## Settings")
    top_n = st.number_input("Top N pairs", min_value=1, max_value=50, value=DEFAULT_TOP_N)
    timeframe = st.selectbox("Indicator timeframe (for ATR / signals)", ["1m", "5m", "15m", "1h", "4h"], index=2)
    refresh = st.number_input("Auto-refresh (seconds)", min_value=10, max_value=600, value=REFRESH_SECONDS)
    notify = st.checkbox("Enable Telegram notifications", value=bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID))
    st.write("---")
    st.write("Environment:")
    st.write(f"- Coinglass API: {'✓ configured' if COINGLASS_API_KEY else '✗ not set'}")
    st.write(f"- Telegram: {'✓ configured' if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else '✗ not set'}")
    manual_refresh = st.button("Refresh Now", key="refresh_button")

with col1:
    placeholder = st.empty()

# State to avoid duplicate notifications
if "last_signals" not in st.session_state:
    st.session_state["last_signals"] = {}

if "last_update" not in st.session_state:
    st.session_state["last_update"] = 0

def main_loop():
    current_time = time.time()
    
    # Check if it's time to refresh
    if current_time - st.session_state["last_update"] >= refresh or manual_refresh:
        st.session_state["last_update"] = current_time
        
        try:
            symbols = get_top_n_pairs_by_volatility(n=top_n, tf=timeframe)
            results = []
            for s in symbols:
                # ensure symbol is in Binance futures format (e.g. BTCUSDT)
                res = generate_recommendation(s, tf=timeframe)
                results.append(res)

            df = pd.DataFrame(results)
            
            # Show table - FIXED: Check if columns exist before displaying
            with placeholder.container():
                st.markdown(f"### Top {len(df)} volatile pairs analysis (updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC)")
                
                # Define expected columns and check which ones exist
                expected_cols = ["symbol", "price", "atr", "atr_pct", "funding", "oi", "cvd", "signal", "rr", "reason", "ts"]
                available_cols = [col for col in expected_cols if col in df.columns]
                
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
                    
                    st.dataframe(display_df, height=420)
                else:
                    st.warning("No data available to display")

                # per-symbol expanders
                for r in results:
                    if not isinstance(r, dict):
                        continue
                        
                    sym = r.get("symbol")
                    signal = r.get("signal", "N/A")
                    price = r.get("price", "N/A")
                    
                    with st.expander(f"{sym} – {signal} – price {price}"):
                        st.write("**Signal:**", signal)
                        st.write("**Reason:**", r.get("reason", "N/A"))
                        st.write("**Price:**", price)
                        st.write("**Funding Rate:**", r.get("funding", "N/A"))
                        st.write("**Open Interest:**", r.get("oi", "N/A"))
                        st.write("**CVD (approx):**", r.get("cvd", "N/A"))
                        st.write("**ATR:**", r.get("atr", "N/A"), "(abs) |", r.get("atr_pct", "N/A"))
                        if r.get("entry"):
                            st.write(f"Entry: {r.get('entry'):.6f}, TP: {r.get('tp'):.6f}, SL: {r.get('sl'):.6f}, RRR: {r.get('rr')}")

                        # plot last 100 candles
                        try:
                            kl = fetch_klines(sym, interval=timeframe, limit=200)
                            fig = go.Figure(data=[go.Candlestick(
                                x=kl["open_time"], open=kl["open"], high=kl["high"], low=kl["low"], close=kl["close"],
                                name="Price"
                            )])
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.write("Chart error:", e)

            # Notifications: compare last_signals to current and send new
            for r in results:
                if not isinstance(r, dict):
                    continue
                    
                s = r.get("symbol")
                sig = r.get("signal")
                last = st.session_state["last_signals"].get(s)
                
                # send when a new actionable signal (SCALP LONG/SHORT) appears or changes
                if sig in ["SCALP LONG", "SCALP SHORT"]:
                    if last != sig:
                        st.session_state["last_signals"][s] = sig
                        text = f"*Signal:* {sig}\n*Pair:* {s}\n*Price:* {r.get('price')}\n*TP:* {r.get('tp')}\n*SL:* {r.get('sl')}\n*RRR:* {r.get('rr')}\n*Reason:* {r.get('reason')}"
                        st.write(f"🔔 New signal for {s}: {sig}")
                        if notify:
                            ok = send_telegram_message(text)
                            st.write("Telegram notified:" , ok)
                else:
                    # clear previous if resolved
                    if st.session_state["last_signals"].get(s) and sig == "WAIT":
                        st.session_state["last_signals"].pop(s, None)

        except Exception as e:
            st.error("Error fetching data: " + str(e))

# Run main loop
main_loop()

# Auto-refresh menggunakan st.rerun()
time.sleep(1)  # Small delay to prevent too rapid refreshes
st.rerun()

st.markdown("---")
st.caption("Notes: This is a starter dashboard. Tune thresholds, add Coinglass endpoints, and backtest rules before live trading. Use testnet and small sizes.")
