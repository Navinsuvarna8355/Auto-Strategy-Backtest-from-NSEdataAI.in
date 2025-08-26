import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- Setup ---
st.set_page_config(page_title="BTC Auto Strategy", layout="wide")
st.title("📊 BTC Strategy Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Strategy Settings")
ma_length = st.sidebar.number_input("Disparity MA Length", min_value=1, max_value=50, value=20)
short_prd = st.sidebar.number_input("Short Period", min_value=1, max_value=20, value=3)
long_prd = st.sidebar.number_input("Long Period", min_value=1, max_value=50, value=6)
threshold = st.sidebar.slider("Signal Threshold (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

# --- Sample BTC Data ---
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    prices = np.cumsum(np.random.randn(len(dates))) + 27500
    return pd.DataFrame({'Date': dates, 'Close': prices})

df = generate_sample_data()
df['MA'] = df['Close'].rolling(window=ma_length).mean()
df['Disparity'] = (df['Close'] - df['MA']) / df['MA'] * 100
df.dropna(inplace=True)

# --- Signal Logic ---
def get_trade_signal(disparity, threshold):
    if disparity > threshold:
        return "Buy CE"
    elif disparity < -threshold:
        return "Buy PE"
    return None

# --- Trade Logger ---
@st.cache_data(show_spinner=False)
def init_logs():
    return []

trade_logs = init_logs()

def log_trade(signal, price, disparity, logs):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    logs.append({
        "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Date": now.date(),
        "Month": now.strftime("%Y-%m"),
        "Trade": signal,
        "Price": round(price, 2),
        "Disparity": round(disparity, 2)
    })
    return logs

# --- Run Strategy ---
st.subheader("🔍 Parsed Strategy Settings")
col1, col2, col3 = st.columns(3)
col1.metric("MA Length", ma_length)
col2.metric("Short Period", short_prd)
col3.metric("Long Period", long_prd)

st.subheader("📈 Disparity Chart")
st.line_chart(df.set_index('Date')[['Close', 'Disparity']])

if st.button("Run Strategy"):
    latest = df.iloc[-1]
    signal = get_trade_signal(latest['Disparity'], threshold)
    if signal:
        trade_logs = log_trade(signal, latest['Close'], latest['Disparity'], trade_logs)
        st.success(f"✅ Auto Trade: {signal} @ {latest['Close']:.2f}")
    else:
        st.warning("No trade signal generated.")

# --- Logs Display ---
st.markdown("---")
st.subheader("📅 Daily Trade Logs")
daily_df = pd.DataFrame(trade_logs)
today = datetime.now(pytz.timezone('Asia/Kolkata')).date()
st.dataframe(daily_df[daily_df['Date'] == today], use_container_width=True)

st.subheader("🗓️ Monthly Trade Logs")
month = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m")
st.dataframe(daily_df[daily_df['Month'] == month], use_container_width=True)
