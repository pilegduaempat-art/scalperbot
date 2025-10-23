import streamlit as st
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Recovery Margin Calculator", layout="centered")

st.title("⚙️ Recovery Margin Calculator (Binance Futures Realtime + Persentase Target)")

st.markdown("""
Masukkan pair (contoh: `BTCUSDT`, `ETHUSDT`, `ALICEUSDT`) untuk mengambil harga **realtime** dari **Binance Futures**.  
Gunakan slider untuk menentukan **persentase target recovery**.
""")

# --- Input PAIR ---
pair = st.text_input("📊 Pair (contoh: BTCUSDT)", value="ALICEUSDT").upper()

# --- Ambil harga realtime dari Binance Futures ---
def get_futures_price(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        st.error(f"Gagal mengambil harga dari Binance: {e}")
        return None

# Tombol ambil harga
if st.button("🔄 Ambil Harga Realtime"):
    current_price_live = get_futures_price(pair)
    if current_price_live:
        st.success(f"Harga terkini {pair}: {current_price_live} USDT")
        st.session_state["current_price_live"] = current_price_live

# Jika belum ada data harga, gunakan default
current_price = st.session_state.get("current_price_live", 0.336)

# --- Input data utama ---
col1, col2 = st.columns(2)
with col1:
    entry_price = st.number_input("🎯 Entry Awal (USDT)", value=0.551, step=0.001, format="%.3f")
    entry_qty = st.number_input("📦 Jumlah Awal (coin)", value=213.0, step=1.0)
    leverage = st.number_input("⚡ Leverage", value=20, step=1)
with col2:
    st.text_input("💰 Harga Sekarang (USDT)", value=f"{current_price:.3f}", disabled=True)
    
    # Slider persentase target
    percent = st.select_slider(
        "📉 Persentase Target Recovery (%)",
        options=[0.5, 1, 2.5, 5, 10, 25, 50, 75, 100],
        value=50
    )

    # Hitung target entry otomatis berdasarkan persentase
    target_entry = current_price + (entry_price - current_price) * (percent / 100)
    st.text_input("🎯 Target Entry (USDT)", value=f"{target_entry:.4f}", disabled=True)

# --- Kalkulasi recovery ---
try:
    numerator = (entry_price * entry_qty) - (target_entry * entry_qty)
    denominator = target_entry - current_price
    add_qty = numerator / denominator if denominator != 0 else 0

    add_position_value = add_qty * current_price
    add_margin = add_position_value / leverage
    total_qty = entry_qty + add_qty
    new_avg_entry = ((entry_price * entry_qty) + (current_price * add_qty)) / total_qty

    # --- Output hasil ---
    st.subheader("📊 Hasil Perhitungan")
    st.write(f"**Persentase Target:** {percent}%")
    st.write(f"**Tambahan Quantity:** {add_qty:,.2f} coin")
    st.write(f"**Nilai Posisi Tambahan:** ${add_position_value:,.2f}")
    st.write(f"**Margin Tambahan (Leverage {leverage}x):** ${add_margin:,.2f}")
    st.write(f"**Harga Entry Baru (Simulasi):** {new_avg_entry:.4f} USDT")

    # --- Visualisasi Chart ---
    st.subheader("📈 Simulasi Harga Recovery")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=["Entry Awal"], y=[entry_price], mode="markers+text",
                             name="Entry Awal", text=[f"{entry_price}"], textposition="top center"))
    fig.add_trace(go.Scatter(x=["Harga Sekarang"], y=[current_price], mode="markers+text",
                             name="Harga Sekarang", text=[f"{current_price}"], textposition="top center"))
    fig.add_trace(go.Scatter(x=["Target Entry"], y=[target_entry], mode="markers+text",
                             name="Target Entry", text=[f"{target_entry}"], textposition="top center"))

    fig.update_layout(
        yaxis_title="Harga (USDT)",
        title=f"Visualisasi Titik Entry vs Target Recovery ({pair})",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Terjadi kesalahan perhitungan: {e}")
