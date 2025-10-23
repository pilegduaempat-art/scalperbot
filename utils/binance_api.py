import requests
import streamlit as st

def get_futures_price(symbol: str):
    """Mengambil harga realtime dari Binance Futures API"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.upper()}"
        response = requests.get(url, timeout=5)
        data = response.json()

        # --- Validasi respon Binance ---
        if "price" in data:
            return float(data["price"])
        elif "msg" in data:
            st.error(f"Binance error: {data['msg']}")
        else:
            st.error(f"Respons tak dikenal dari Binance: {data}")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal menghubungi Binance: {e}")
        return None
    except ValueError:
        st.error("Gagal memproses data harga dari Binance.")
        return None
