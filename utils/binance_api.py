import requests
import streamlit as st

def get_futures_price(symbol: str):
    """Fallback ke Binance Spot API jika Futures diblokir"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "price" in data:
            return float(data["price"])
        else:
            st.error(f"Fallback gagal: {data}")
        return None
    except Exception as e:
        st.error(f"Gagal mengambil harga (fallback): {e}")
        return None

