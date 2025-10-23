import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Recovery Margin Calculator", layout="centered")

st.title("⚙️ Recovery Margin Calculator (Crypto Futures)")

st.markdown("""
Hitung berapa tambahan margin (USDT) yang dibutuhkan agar **average entry** turun ke harga target.
""")

# Input data utama
col1, col2 = st.columns(2)
with col1:
    entry_price = st.number_input("🎯 Entry Awal (USDT)", value=0.551, step=0.001, format="%.3f")
    entry_qty = st.number_input("📦 Jumlah Awal (coin)", value=213.0, step=1.0)
    leverage = st.number_input("⚡ Leverage", value=20, step=1)
with col2:
    current_price = st.number_input("💰 Harga Sekarang (USDT)", value=0.336, step=0.001, format="%.3f")
    target_entry = st.number_input("🎯 Target Entry (USDT)", value=0.338, step=0.001, format="%.3f")

# Kalkulasi
try:
    numerator = (entry_price * entry_qty) - (target_entry * entry_qty)
    denominator = target_entry - current_price
    add_qty = numerator / denominator if denominator != 0 else 0

    add_position_value = add_qty * current_price
    add_margin = add_position_value / leverage
    total_qty = entry_qty + add_qty
    new_avg_entry = ((entry_price * entry_qty) + (current_price * add_qty)) / total_qty

    # Output hasil
    st.subheader("📊 Hasil Perhitungan")
    st.write(f"**Tambahan Quantity:** {add_qty:,.2f} coin")
    st.write(f"**Nilai Posisi Tambahan:** ${add_position_value:,.2f}")
    st.write(f"**Margin Tambahan (Leverage {leverage}x):** ${add_margin:,.2f}")
    st.write(f"**Harga Entry Baru (Simulasi):** {new_avg_entry:.4f} USDT")

    # Visualisasi Chart
    st.subheader("📈 Simulasi Harga Recovery")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=["Entry Awal"], y=[entry_price], mode="markers+text",
                             name="Entry Awal", text=[f"{entry_price}"], textposition="top center"))
    fig.add_trace(go.Scatter(x=["Harga Sekarang"], y=[current_price], mode="markers+text",
                             name="Harga Sekarang", text=[f"{current_price}"], textposition="top center"))
    fig.add_trace(go.Scatter(x=["Target Entry"], y=[target_entry], mode="markers+text",
                             name="Target Entry", text=[f"{target_entry}"], textposition="top center"))

    fig.update_layout(yaxis_title="Harga (USDT)",
                      title="Visualisasi Titik Entry vs Target Recovery",
                      template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Terjadi kesalahan perhitungan: {e}")
