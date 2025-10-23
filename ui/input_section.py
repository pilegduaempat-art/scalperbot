import streamlit as st

def get_user_inputs(get_futures_price_func):
    """Bagian input data pengguna dan pengambilan harga realtime"""
    pair = st.text_input("📊 Pair (contoh: BTCUSDT)", value="ALICEUSDT").upper()

    if st.button("🔄 Ambil Harga Realtime"):
        current_price_live = get_futures_price_func(pair)
        if current_price_live:
            st.success(f"Harga terkini {pair}: {current_price_live} USDT")
            st.session_state["current_price_live"] = current_price_live

    current_price = st.session_state.get("current_price_live", 0.336)

    col1, col2 = st.columns(2)
    with col1:
        entry_price = st.number_input("🎯 Entry Awal (USDT)", value=0.551, step=0.001, format="%.3f")
        entry_qty = st.number_input("📦 Jumlah Awal (coin)", value=213.0, step=1.0)
        leverage = st.number_input("⚡ Leverage", value=20, step=1)

    with col2:
        st.text_input("💰 Harga Sekarang (USDT)", value=f"{current_price:.3f}", disabled=True)
        percent = st.select_slider(
            "📉 Persentase Target Recovery (%)",
            options=[0.5, 1, 2.5, 5, 10, 25, 50, 75, 100],
            value=50
        )
        target_entry = current_price + (entry_price - current_price) * (percent / 100)
        st.text_input("🎯 Target Entry (USDT)", value=f"{target_entry:.4f}", disabled=True)

    return pair, current_price, entry_price, entry_qty, leverage, percent, target_entry
