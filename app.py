import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page setup
st.set_page_config(page_title="BTC Disparity Strategy", layout="wide")
st.title("📊 BTC Strategy Dashboard")

# --- Sidebar: Separate Inputs ---
st.sidebar.header("⚙️ Strategy Settings")
ma_length = st.sidebar.number_input("Disparity MA Length", min_value=1, max_value=50, value=14)
short_prd = st.sidebar.number_input("Short Period", min_value=1, max_value=20, value=5)
long_prd = st.sidebar.number_input("Long Period", min_value=1, max_value=50, value=20)

# --- Sample BTC Data Generator ---
def generate_sample_data():
    np.random.seed(42)
    now = datetime.now()
    dates = pd.date_range(end=now, periods=100, freq='5min')
    prices = np.cumsum(np.random.randn(len(dates))) + 27500
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    return df

df = generate_sample_data()

# --- Disparity Index Calculation ---
df['MA'] = df['Close'].rolling(window=ma_length).mean()
df['Disparity'] = (df['Close'] - df['MA']) / df['MA'] * 100
df = df.dropna()

# --- Display Parsed Settings ---
st.subheader("🔍 Strategy Settings")
col1, col2, col3 = st.columns(3)
col1.metric("MA Length", ma_length)
col2.metric("Short Period", short_prd)
col3.metric("Long Period", long_prd)

# --- Chart (2 Lines Only) ---
st.subheader("📈 Disparity Index Chart")
chart_df = df[['Date', 'Close', 'Disparity']].set_index('Date')
st.line_chart(chart_df)

# --- Signal Logic (Simple) ---
st.markdown("---")
if st.button("Run Strategy"):
    signal = "BUY" if short_prd > long_prd else "SELL"
    st.success(f"Signal: **{signal}** | MA={ma_length}, Short={short_prd}, Long={long_prd}")
