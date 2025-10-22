# app.py
"""
Streamlit dashboard + Telegram notifier for automated analysis of top volatile Binance Futures pairs.
Features:
- Fetch top N volatile pairs (by 24h price change or ATR) using WebSocket realtime data
- Calculate ATR, approximate CVD (from aggTrades), fetch Open Interest & Funding Rate
- Generate recommendation signals (Scalp Long / Scalp Short / Wait)
- Streamlit UI with charts and auto-refresh
- Telegram notifications for new signals
Requirements:
pip install streamlit pandas requests plotly python-dotenv websocket-client
"""

import os
import time
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import json
import threading
from datetime import datetime, timedelta
import websocket

# CONFIG / ENV
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_WS_BASE = "wss://fstream.binance.com/ws"
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")  # optional
TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")  # optional
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")  # optional

# App defaults
DEFAULT_TOP_N = 10
DEFAULT_TIMEFRAME = "15m"  # used for ATR/ohlcv
REFRESH_SECONDS = 60

# Global variables for WebSocket data
price_data = {}
klines_data = {}
ws_connections = {}

# -------------------------
# WebSocket Functions
# -------------------------
def on_message(ws, message):
    """Handle incoming WebSocket messages"""
    try:
        data = json.loads(message)
        
        if 'e' in data:
            # Kline data
            if data['e'] == 'kline':
                symbol = data['s']
                kline = data['k']
                if symbol not in klines_data:
                    klines_data[symbol] = []
                
                # Update klines data
                kline_info = {
                    'timestamp': data['E'],
                    'open_time': kline['t'],
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'close_time': kline['T'],
                    'num_trades': kline['n']
                }
                
                # Keep only last 100 klines per symbol
                if len(klines_data[symbol]) >= 100:
                    klines_data[symbol].pop(0)
                klines_data[symbol].append(kline_info)
                
            # Ticker data for realtime prices
            elif data['e'] == '24hrTicker':
                symbol = data['s']
                price_data[symbol] = {
                    'price': float(data['c']),
                    'price_change_percent': float(data['P']),
                    'volume': float(data['v']),
                    'timestamp': data['E']
                }
                
    except Exception as e:
        print(f"WebSocket message error: {e}")

def on_error(ws, error):
    """Handle WebSocket errors"""
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Handle WebSocket closure"""
    print("WebSocket connection closed")

def on_open(ws):
    """Handle WebSocket opening"""
    print("WebSocket connection opened")

def start_websocket_connection(symbols):
    """Start WebSocket connections for given symbols"""
    global ws_connections
    
    # Stop existing connections
    for symbol, ws in ws_connections.items():
        try:
            ws.close()
        except:
            pass
    
    ws_connections.clear()
    
    # Start new connections for each symbol
    for symbol in symbols:
        try:
            # Kline stream
            kline_stream = f"{symbol.lower()}@kline_{DEFAULT_TIMEFRAME}"
            ws_kline = websocket.WebSocketApp(
                f"{BINANCE_WS_BASE}/{kline_stream}",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws_kline.on_open = on_open
            
            # Ticker stream for realtime prices
            ticker_stream = f"{symbol.lower()}@ticker"
            ws_ticker = websocket.WebSocketApp(
                f"{BINANCE_WS_BASE}/{ticker_stream}",
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws_ticker.on_open = on_open
            
            # Start WebSocket in separate threads
            def run_kline_ws():
                ws_kline.run_forever()
            
            def run_ticker_ws():
                ws_ticker.run_forever()
            
            kline_thread = threading.Thread(target=run_kline_ws, daemon=True)
            ticker_thread = threading.Thread(target=run_ticker_ws, daemon=True)
            
            kline_thread.start()
            ticker_thread.start()
            
            ws_connections[symbol] = {
                'kline': ws_kline,
                'ticker': ws_ticker,
                'kline_thread': kline_thread,
                'ticker_thread': ticker_thread
            }
            
            time.sleep(0.1)  # Small delay between connections
            
        except Exception as e:
            print(f"Error starting WebSocket for {symbol}: {e}")

# -------------------------
# Helper: Binance REST calls (for initial data and fallback)
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
    """Fallback REST API for klines data"""
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
    if not COINGLASS_API_KEY:
        return None
    headers = {"coinglassSecret": COINGLASS_API_KEY}
    try:
        url = f"https://openapi.coinglass.com/public/v2/futures/{metric}"
        r = requests.get(url, headers=headers, params={"symbol": symbol})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# -------------------------
# Indicators & Signals
# -------------------------
def atr_from_websocket(symbol: str, period: int = 14):
    """Calculate ATR from WebSocket klines data"""
    try:
        if symbol not in klines_data or len(klines_data[symbol]) < period:
            return 0.0
        
        klines = klines_data[symbol]
        high_low = [k['high'] - k['low'] for k in klines]
        high_close = [abs(k['high'] - klines[i-1]['close']) if i > 0 else 0 for i, k in enumerate(klines)]
        low_close = [abs(k['low'] - klines[i-1]['close']) if i > 0 else 0 for i, k in enumerate(klines)]
        
        tr = [max(high_low[i], high_close[i], low_close[i]) for i in range(len(klines))]
        atr_value = np.mean(tr[-period:])
        return atr_value
    except Exception:
        return 0.0

def get_realtime_price(symbol: str):
    """Get realtime price from WebSocket data"""
    if symbol in price_data:
        return price_data[symbol]['price']
    else:
        # Fallback to REST API
        try:
            ticker = fetch_24hr_ticker()
            for t in ticker:
                if t['symbol'] == symbol:
                    return float(t['lastPrice'])
        except:
            pass
    return 0.0

def approximate_cvd_from_aggtrades(symbol: str, minutes: int = 60):
    try:
        df = fetch_klines(symbol, interval="1m", limit=minutes)
        df["taker_buy_base"] = df["taker_buy_base"].astype(float)
        df["delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
        cvd = df["delta"].cumsum().iloc[-1]
        return cvd
    except Exception:
        return None

def generate_recommendation(symbol: str, tf: str = "15m"):
    """Generate recommendation using realtime WebSocket data"""
    try:
        # Get realtime price from WebSocket
        price = get_realtime_price(symbol)
        if price == 0:
            return {"symbol": symbol, "error": "No price data available"}
        
        # Calculate ATR from WebSocket klines
        recent_atr = atr_from_websocket(symbol)
        
        # Fetch other data via REST API
        funding = fetch_funding_rate(symbol)
        oi = fetch_open_interest(symbol)
        cvd = approximate_cvd_from_aggtrades(symbol, minutes=60)

        # Basic thresholds (tuneable)
        atr_pct = recent_atr / price if price else 0.0

        signal = "WAIT"
        reason = []
        entry = None
        tp = None
        sl = None
        rr = None

        # Scalp Long logic:
        if funding < -0.0005 and (cvd is not None and cvd < 0):
            signal = "SCALP LONG"
            entry = price
            tp = price * (1 + max(0.01, min(0.05, atr_pct * 5)))
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
            "ts": datetime.utcnow().isoformat(),
            "data_source": "WebSocket" if symbol in price_data else "REST API"
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# -------------------------
# Utilities
# -------------------------
def get_top_n_pairs_by_volatility(n=10, tf="15m"):
    try:
        tickers = fetch_24hr_ticker()
        fut = [t for t in tickers if t["symbol"].endswith("USDT") and ("PERP" not in t)]
        df = pd.DataFrame(fut)
        df["priceChangePercent"] = df["priceChangePercent"].astype(float)
        df["abs_change"] = df["priceChangePercent"].abs()
        top = df.sort_values("abs_change", ascending=False).head(n)
        return top["symbol"].tolist()
    except Exception:
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
st.info("🔴 **REALTIME MODE**: Using WebSocket for realtime price data")

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
    st.write(f"- WebSocket Connections: {len(ws_connections)}")
    manual_refresh = st.button("Refresh Now", key="refresh_button")
    
    # WebSocket status
    st.write("---")
    st.write("WebSocket Status:")
    if price_data:
        st.success("🟢 Connected - Real-time data")
        st.write(f"Tracking {len(price_data)} symbols")
    else:
        st.warning("🟡 Connecting...")

with col1:
    placeholder = st.empty()

# State to avoid duplicate notifications
if "last_signals" not in st.session_state:
    st.session_state["last_signals"] = {}

if "last_update" not in st.session_state:
    st.session_state["last_update"] = 0

if "websocket_started" not in st.session_state:
    st.session_state["websocket_started"] = False

def main_loop():
    current_time = time.time()
    
    # Check if it's time to refresh
    if current_time - st.session_state["last_update"] >= refresh or manual_refresh:
        st.session_state["last_update"] = current_time
        
        try:
            symbols = get_top_n_pairs_by_volatility(n=top_n, tf=timeframe)
            
            # Start WebSocket connections if not already started
            if not st.session_state["websocket_started"] or manual_refresh:
                start_websocket_connection(symbols)
                st.session_state["websocket_started"] = True
                st.toast("🔄 WebSocket connections started", icon="✅")
            
            results = []
            for s in symbols:
                res = generate_recommendation(s, tf=timeframe)
                results.append(res)

            df = pd.DataFrame(results)
            
            # Show table
            with placeholder.container():
                st.markdown(f"### Top {len(df)} volatile pairs analysis (updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC)")
                st.caption("🟢 Real-time WebSocket data | 🟡 REST API fallback")
                
                # Define expected columns and check which ones exist
                expected_cols = ["symbol", "price", "atr", "atr_pct", "funding", "oi", "cvd", "signal", "rr", "reason", "ts", "data_source"]
                available_cols = [col for col in expected_cols if col in df.columns]
                
                if available_cols:
                    display_df = df[available_cols].copy()
                    
                    # Format numeric columns if they exist
                    if "price" in display_df.columns:
                        display_df["price"] = display_df["price"].map(lambda x: f"${x:.6f}" if pd.notnull(x) else x)
                    if "atr" in display_df.columns:
                        display_df["atr"] = display_df["atr"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else x)
                    if "atr_pct" in display_df.columns:
                        display_df["atr_pct"] = display_df["atr_pct"].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else x)
                    if "funding" in display_df.columns:
                        display_df["funding"] = display_df["funding"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else x)
                    if "oi" in display_df.columns:
                        display_df["oi"] = display_df["oi"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else x)
                    
                    # Color coding for data source
                    def highlight_row(row):
                        if 'data_source' in row and row['data_source'] == 'WebSocket':
                            return ['background-color: #d4edda'] * len(row)
                        else:
                            return ['background-color: #fff3cd'] * len(row)
                    
                    if 'data_source' in display_df.columns:
                        styled_df = display_df.style.apply(highlight_row, axis=1)
                    else:
                        styled_df = display_df
                    
                    st.dataframe(styled_df, height=420)
                else:
                    st.warning("No data available to display")

                # per-symbol expanders
                for r in results:
                    if not isinstance(r, dict):
                        continue
                        
                    sym = r.get("symbol")
                    signal = r.get("signal", "N/A")
                    price = r.get("price", "N/A")
                    data_source = r.get("data_source", "REST API")
                    
                    with st.expander(f"{sym} – {signal} – ${price} – {data_source}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📊 Signal Information**")
                            st.write(f"**Signal:** {signal}")
                            st.write(f"**Price:** ${price}")
                            st.write(f"**Funding Rate:** {r.get('funding', 'N/A')}")
                            st.write(f"**Open Interest:** {r.get('oi', 'N/A')}")
                            st.write(f"**Data Source:** {data_source}")
                            
                        with col2:
                            st.write("**📈 Technical Details**")
                            st.write(f"**CVD (approx):** {r.get('cvd', 'N/A')}")
                            st.write(f"**ATR:** {r.get('atr', 'N/A')} ({r.get('atr_pct', 'N/A')})")
                            if r.get("entry"):
                                st.write(f"**Entry:** ${r.get('entry'):.6f}")
                                st.write(f"**TP:** ${r.get('tp'):.6f}")
                                st.write(f"**SL:** ${r.get('sl'):.6f}")
                                st.write(f"**Risk/Reward:** {r.get('rr')}")
                        
                        st.write("**💡 Reason:**", r.get("reason", "N/A"))

                        # Real-time price chart from WebSocket data
                        try:
                            if sym in klines_data and klines_data[sym]:
                                klines = klines_data[sym]
                                times = [datetime.fromtimestamp(k['open_time']/1000) for k in klines]
                                opens = [k['open'] for k in klines]
                                highs = [k['high'] for k in klines]
                                lows = [k['low'] for k in klines]
                                closes = [k['close'] for k in klines]
                                
                                fig = go.Figure(data=[go.Candlestick(
                                    x=times, open=opens, high=highs, low=lows, close=closes,
                                    name="Price"
                                )])
                                fig.update_layout(
                                    height=300, 
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    title=f"Real-time {sym} Price (WebSocket)",
                                    xaxis_title="Time",
                                    yaxis_title="Price"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Fallback to REST API chart
                                kl = fetch_klines(sym, interval=timeframe, limit=50)
                                fig = go.Figure(data=[go.Candlestick(
                                    x=kl["open_time"], open=kl["open"], high=kl["high"], low=kl["low"], close=kl["close"],
                                    name="Price"
                                )])
                                fig.update_layout(
                                    height=300, 
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    title=f"{sym} Price (REST API)",
                                    xaxis_title="Time",
                                    yaxis_title="Price"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.write("Chart error:", e)

            # Notifications
            for r in results:
                if not isinstance(r, dict):
                    continue
                    
                s = r.get("symbol")
                sig = r.get("signal")
                last = st.session_state["last_signals"].get(s)
                
                if sig in ["SCALP LONG", "SCALP SHORT"]:
                    if last != sig:
                        st.session_state["last_signals"][s] = sig
                        text = f"🔔 *Real-time Signal Alert*\n\n*Signal:* {sig}\n*Pair:* {s}\n*Price:* ${r.get('price')}\n*Entry:* ${r.get('entry'):.6f}\n*TP:* ${r.get('tp'):.6f}\n*SL:* ${r.get('sl'):.6f}\n*RRR:* {r.get('rr')}\n*Reason:* {r.get('reason')}\n*Data Source:* {r.get('data_source', 'Real-time')}"
                        st.success(f"🔔 New real-time signal for {s}: {sig}")
                        if notify:
                            ok = send_telegram_message(text)
                            if ok:
                                st.toast(f"✅ Telegram alert sent for {s}", icon="✅")
                else:
                    if st.session_state["last_signals"].get(s) and sig == "WAIT":
                        st.session_state["last_signals"].pop(s, None)

        except Exception as e:
            st.error("Error fetching data: " + str(e))

# Run main loop
main_loop()

# Auto-refresh
time.sleep(1)
st.rerun()

st.markdown("---")
st.caption("""
**Real-time Features:**
- WebSocket connections for real-time price data
- Real-time candlestick charts
- Instant signal generation
- Fallback to REST API when WebSocket fails

**Notes:** This is a real-time dashboard. Tune thresholds and backtest rules before live trading. Use testnet and small sizes.
""")
