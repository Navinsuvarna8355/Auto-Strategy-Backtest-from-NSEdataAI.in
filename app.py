import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="BTC Auto Strategy", layout="wide")
st.title("📊 BTC Strategy Dashboard")

# --- Persistent Strategy Settings ---
if "ma_length" not in st.session_state:
    st.session_state.ma_length = 20
if "short_prd" not in st.session_state:
    st.session_state.short_prd = 3
if "long_prd" not in st.session_state:
    st.session_state.long_prd = 6
if "threshold" not in st.session_state:
    st.session_state.threshold = 1.5
if "trade_logs" not in st.session_state:
    st.session_state.trade_logs = []

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Strategy Settings")
st.session_state.ma_length = st.sidebar.number_input("Disparity MA Length", min_value=1, max_value=50, value=st.session_state.ma_length)
st.session_state.short_prd = st.sidebar.number_input("Short Period", min_value=1, max_value=20, value=st.session_state.short_prd)
st.session_state.long_prd = st.sidebar.number_input("Long Period", min_value=1, max_value=50, value=st.session_state.long_prd)
st.session_state.threshold = st.sidebar.slider("Signal Threshold (%)", min_value=0.5, max_value=5.0, value=st.session_state.threshold, step=0.1)

# --- Sample BTC Data ---
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    initial_price = 27500
    volatility = 50 
    prices = initial_price + np.cumsum(np.random.randn(len(dates)) * volatility)
    for i in range(5):
        jump_idx = np.random.randint(10, 90)
        jump_size = np.random.uniform(-500, 500)
        prices[jump_idx:] += jump_size
    return pd.DataFrame({'Date': dates, 'Close': prices})

df = generate_sample_data()
df['MA'] = df['Close'].rolling(window=st.session_state.ma_length).mean()
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
def log_trade(signal, price, disparity):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    st.session_state.trade_logs.append({
        "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Date": str(now.date()),
        "Month": now.strftime("%Y-%m"),
        "Trade": signal,
        "Price": round(price, 2),
        "Disparity": round(disparity, 2)
    })

# --- Parsed Settings Display ---
st.subheader("🔍 Parsed Strategy Settings")
col1, col2, col3 = st.columns(3)
col1.metric("MA Length", st.session_state.ma_length)
col2.metric("Short Period", st.session_state.short_prd)
col3.metric("Long Period", st.session_state.long_prd)

# --- Interactive Plotly Chart ---
st.subheader("📈 Interactive BTC Price Chart")
fig = px.line(df, x='Date', y='Close', title='BTC Price and Moving Average')
fig.add_scatter(x=df['Date'], y=df['MA'], name='Moving Average', mode='lines')
st.plotly_chart(fig, use_container_width=True)

# --- Strategy Toggle ---
auto_mode = st.toggle("🔄 Auto Strategy Mode", value=False)

if auto_mode:
    latest = df.iloc[-1]
    signal = get_trade_signal(latest['Disparity'], st.session_state.threshold)
    if signal:
        log_trade(signal, latest['Close'], latest['Disparity'])
        st.success(f"✅ Auto Trade: {signal} @ {latest['Close']:.2f}")
    else:
        st.info("No trade signal at this moment.")

# --- Logs Display ---
st.markdown("---")
daily_df = pd.DataFrame(st.session_state.trade_logs)
today = str(datetime.now(pytz.timezone("Asia/Kolkata")).date())
month = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m")

if not daily_df.empty and all(col in daily_df.columns for col in ['Date', 'Month']):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Daily Trade Logs")
        st.dataframe(daily_df[daily_df['Date'] == today], use_container_width=True)
    with col2:
        st.subheader("🗓️ Monthly Trade Logs")
        st.dataframe(daily_df[daily_df['Month'] == month], use_container_width=True)
else:
    st.info("📭 No trades logged yet. Toggle strategy ON to begin auto-trading.")
