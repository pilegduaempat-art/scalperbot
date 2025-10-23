import requests
import streamlit as st

def get_futures_price(symbol: str):
    """Mengambil harga realtime dari Binance Futures API"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        st.error(f"Gagal mengambil harga dari Binance: {e}")
        return None
