import streamlit as st

# ========================================
# 1️⃣ Environment Check Block
# ========================================
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import plotly
    import pytz
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module: {e.name}")
    st.stop()
else:
    st.success("✅ All core modules found")
    st.write({
        "streamlit": st.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "yfinance": yf.__version__,
        "plotly": plotly.__version__,
        "pytz": pytz.__version__,
    })

# ========================================
# 2️⃣ Dashboard Header
# ========================================
st.title("📊 Dual‑Symbol Trading Dashboard")
st.caption("Live NIFTY / BANKNIFTY View with Env‑Check")

# ========================================
# 3️⃣ Sidebar Configurations
# ========================================
st.sidebar.header("Settings")
symbol_1 = st.sidebar.text_input("Symbol 1", value="^NSEI")   # NIFTY
symbol_2 = st.sidebar.text_input("Symbol 2", value="^NSEBANK")  # BANKNIFTY
period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])

# ========================================
# 4️⃣ Data Fetch Function
# ========================================
@st.cache_data(ttl=300)
def get_data(sym, per, inter):
    try:
        data = yf.download(sym, period=per, interval=inter)
        return data
    except Exception as e:
        st.error(f"Data fetch error for {sym}: {e}")
        return None

# ========================================
# 5️⃣ Live Data Panels
# ========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader(symbol_1)
    df1 = get_data(symbol_1, period, interval)
    if df1 is not None and not df1.empty:
        st.dataframe(df1.tail())
    else:
        st.warning("No data retrieved.")

with col2:
    st.subheader(symbol_2)
    df2 = get_data(symbol_2, period, interval)
    if df2 is not None and not df2.empty:
        st.dataframe(df2.tail())
    else:
        st.warning("No data retrieved.")

# ========================================
# 6️⃣ Footer Notes
# ========================================
st.markdown("---")
st.caption("Built for rapid Streamlit Cloud verification & NIFTY/BANKNIFTY live checks.")
