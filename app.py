import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from strategy import apply_disparity_index, generate_signals, simulate_trades
from fetcher import get_ohlc
from utils import breakdown_pnl, export_csv

st.set_page_config(layout="wide")
st.title("📈 Disparity Index Dashboard")

# --- Sidebar Inputs ---
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
di_period = st.sidebar.slider("DI Period", 5, 50, 14)
di_threshold = st.sidebar.slider("DI Threshold (%)", 0.5, 5.0, 2.0)

# --- Fetch & Process Data ---
df = get_ohlc(symbol)
df = apply_disparity_index(df, di_period)
df = generate_signals(df, di_threshold)
trade_log = simulate_trades(df, symbol)

# --- Display PnL ---
monthly_pnl, daily_pnl = breakdown_pnl(trade_log)
st.subheader("📅 Monthly PnL")
st.dataframe(monthly_pnl)
st.subheader("📆 Daily PnL")
st.dataframe(daily_pnl)

# --- Trade Log & Export ---
st.subheader("📜 Trade Log")
st.dataframe(trade_log)
export_csv(trade_log, f"{symbol}_trade_log.csv")

# --- Plotly Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DI_short'], mode='lines', name='DI Short', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DI_long'], mode='lines', name='DI Long', line=dict(color='green')))

for _, trade in trade_log.iterrows():
    fig.add_trace(go.Scatter(x=[trade['Entry Time']], y=[trade['Entry Price']],
                             mode='markers', name='Buy', marker=dict(color='lime', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=[trade['Exit Time']], y=[trade['Exit Price']],
                             mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))

fig.update_layout(title=f"{symbol} Disparity Index Strategy",
                  xaxis_title="Time (IST)", yaxis_title="DI Value (%)",
                  template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)
