import requests
import streamlit as st

def get_futures_price(symbol: str):
    try:
        proxy_url = f"https://your-proxy-domain.com/futures_price?symbol={symbol.upper()}"
        response = requests.get(proxy_url, timeout=5)
        data = response.json()
        if "price" in data:
            return float(data["price"])
        elif "msg" in data:
            st.error(f"Binance error: {data['msg']}")
        else:
            st.error(f"Respons tak dikenal: {data}")
        return None
    except Exception as e:
        st.error(f"Gagal menghubungi proxy: {e}")
        return None


