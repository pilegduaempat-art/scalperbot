import streamlit as st
from utils.binance_api import get_futures_price
from utils.calculator import calculate_recovery
from utils.chart import plot_recovery_chart
from ui.layout import set_layout
from ui.input_section import get_user_inputs

# --- Setup halaman utama ---
set_layout()

st.title("⚙️ Recovery Margin Calculator (Binance Futures Realtime + Persentase Target)")

st.markdown("""
Masukkan pair (contoh: `BTCUSDT`, `ETHUSDT`, `ALICEUSDT`) untuk mengambil harga **realtime** dari **Binance Futures**.  
Gunakan slider untuk menentukan **persentase target recovery**.
""")

# --- Input utama ---
pair, current_price, entry_price, entry_qty, leverage, percent, target_entry = get_user_inputs(get_futures_price)

# --- Kalkulasi hasil ---
try:
    results = calculate_recovery(entry_price, entry_qty, leverage, current_price, target_entry)
    st.subheader("📊 Hasil Perhitungan")
    st.write(f"**Persentase Target:** {percent}%")
    st.write(f"**Tambahan Quantity:** {results['add_qty']:,.2f} coin")
    st.write(f"**Nilai Posisi Tambahan:** ${results['add_position_value']:,.2f}")
    st.write(f"**Margin Tambahan (Leverage {leverage}x):** ${results['add_margin']:,.2f}")
    st.write(f"**Harga Entry Baru (Simulasi):** {results['new_avg_entry']:.4f} USDT")

    # --- Visualisasi Chart ---
    st.subheader("📈 Simulasi Harga Recovery")
    fig = plot_recovery_chart(pair, entry_price, current_price, target_entry)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Terjadi kesalahan perhitungan: {e}")
