import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
import plotly.graph_objects as go
import json
import os

# --- Page Setup ---
st.set_page_config(page_title="BTC Auto Strategy", layout="wide")
st.title("📊 BTC Strategy Dashboard")

# --- Function to save settings ---
def save_settings():
    settings = {
        "ma_length": st.session_state.ma_length,
        "short_prd": st.session_state.short_prd,
        "long_prd": st.session_state.long_prd,
        "threshold": st.session_state.threshold
    }
    with open("settings.json", "w") as f:
        json.dump(settings, f)
    st.success("Settings saved successfully!")

# --- Function to load settings ---
def load_settings():
    if os.path.exists("settings.json"):
        with open("settings.json", "r") as f:
            settings = json.load(f)
        st.session_state.ma_length = settings["ma_length"]
        st.session_state.short_prd = settings["short_prd"]
        st.session_state.long_prd = settings["long_prd"]
        st.session_state.threshold = settings["threshold"]

# --- Persistent Strategy Settings (with load on startup) ---
if "ma_length" not in st.session_state:
    st.session_state.ma_length = 20
    load_settings()
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

# --- Save Settings Button ---
if st.sidebar.button("💾 Save Settings"):
    save_settings()

# --- Sample BTC Data for Candlestick Chart ---
def generate_sample_data():
    np.random.seed(42)
    start_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    end_time = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    time_diff = end_time - start_time
    num_periods = int(time_diff.total_seconds() / 300)
    
    dates = pd.date_range(start=start_time, periods=num_periods, freq='5min')
    
    prices = 27500 + np.cumsum(np.random.randn(num_periods) * 50)
    
    # Create OHLC data
    ohlc = pd.DataFrame(index=dates, columns=['Open', 'High', 'Low', 'Close'])
    ohlc['Open'] = prices
    ohlc['Close'] = ohlc['Open'] + np.random.randn(num_periods) * 25
    ohlc['High'] = ohlc[['Open', 'Close']].max(axis=1) + np.abs(np.random.randn(num_periods) * 10)
    ohlc['Low'] = ohlc[['Open', 'Close']].min(axis=1) - np.abs(np.random.randn(num_periods) * 10)
    
    return ohlc.reset_index().rename(columns={'index': 'Date'})

df = generate_sample_data()
df['MA'] = df['Close'].rolling(window=st.session_state.ma_length).mean()
df['Disparity'] = (df['Close'] - df['MA']) / df['MA'] * 100
df.dropna(inplace=True)

# --- Signal Logic ---
def get_trade_signal(disparity, threshold):
    if disparity > threshold:
        return "Buy PE"
    elif disparity < -threshold:
        return "Buy CE"
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

# --- Interactive Plotly Candlestick Chart ---
st.subheader("📈 Interactive Candlestick Chart with Signals")
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

# Add MA line
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA'], mode='lines', name='MA', line=dict(color='orange', width=2)))

# Backtest button
if st.button("▶️ Run Backtest"):
    st.session_state.trade_logs = []
    buy_ce_signals = []
    buy_pe_signals = []
    for index, row in df.iterrows():
        signal = get_trade_signal(row['Disparity'], st.session_state.threshold)
        if signal == "Buy CE":
            log_trade(signal, row['Close'], row['Disparity'])
            buy_ce_signals.append((row['Date'], row['Low'] - 100))
        elif signal == "Buy PE":
            log_trade(signal, row['Close'], row['Disparity'])
            buy_pe_signals.append((row['Date'], row['High'] + 100))

    # Add Buy CE signals to the chart
    if buy_ce_signals:
        fig.add_trace(go.Scatter(
            x=[x[0] for x in buy_ce_signals],
            y=[x[1] for x in buy_ce_signals],
            mode='markers',
            name='Buy CE',
            marker=dict(color='green', size=15, symbol='triangle-up'),
            text=["Buy CE"] * len(buy_ce_signals)
        ))

    # Add Buy PE signals to the chart
    if buy_pe_signals:
        fig.add_trace(go.Scatter(
            x=[x[0] for x in buy_pe_signals],
            y=[x[1] for x in buy_pe_signals],
            mode='markers',
            name='Buy PE',
            marker=dict(color='red', size=15, symbol='triangle-down'),
            text=["Buy PE"] * len(buy_pe_signals)
        ))
    
    st.success("Backtest completed! Signals are plotted on the chart.")

st.plotly_chart(fig, use_container_width=True)

# --- Logs Display ---
st.markdown("---")
daily_df = pd.DataFrame(st.session_state.trade_logs)
if not daily_df.empty:
    today = str(datetime.now(pytz.timezone("Asia/Kolkata")).date())
    month = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Daily Trade Logs")
        st.dataframe(daily_df[daily_df['Date'] == today], use_container_width=True)
    with col2:
        st.subheader("🗓️ Monthly Trade Logs")
        st.dataframe(daily_df[daily_df['Month'] == month], use_container_width=True)
else:
    st.info("📭 Click 'Run Backtest' to see trade signals.")
