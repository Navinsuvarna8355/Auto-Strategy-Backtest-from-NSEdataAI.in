import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Config ---
st.set_page_config(layout="wide")
st.title("📊 NIFTY/BANKNIFTY Disparity Index Dashboard")

# --- Strategy Settings ---
st.sidebar.header("Strategy Settings")
di_period = st.sidebar.slider("Disparity Index Period", 5, 50, 14)
di_threshold = st.sidebar.slider("DI Threshold (%)", 0.5, 5.0, 2.0)

# --- Symbol Selection ---
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])

# --- Simulated OHLC Fetcher (Replace with live feed) ---
def get_ohlc(symbol):
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(60)][::-1]
    prices = np.linspace(19500, 19600, 60) if symbol == "NIFTY" else np.linspace(44500, 44650, 60)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices + np.random.normal(0, 10, 60)
    })
    return df

df = get_ohlc(symbol)
df['DI'] = ((df['close'] - df['close'].rolling(di_period).mean()) / df['close'].rolling(di_period).mean()) * 100
df['DI_short'] = df['DI']
df['DI_long'] = df['DI'].rolling(5).mean()

# --- Signal Logic ---
df['Signal'] = np.where(df['DI_short'] > di_threshold, 'Buy',
                 np.where(df['DI_short'] < -di_threshold, 'Sell', 'Hold'))

# --- Trade Simulation ---
trades = []
position = None
entry_price = 0

for i in range(1, len(df)):
    signal = df.loc[i, 'Signal']
    price = df.loc[i, 'close']
    time = df.loc[i, 'timestamp'].strftime('%Y-%m-%d %H:%M:%S')

    if signal == 'Buy' and position is None:
        position = 'Long'
        entry_price = price
        entry_time = time
    elif signal == 'Sell' and position == 'Long':
        pnl = price - entry_price
        trades.append({
            "Symbol": symbol,
            "Entry Time": entry_time,
            "Entry Price": round(entry_price, 2),
            "Exit Time": time,
            "Exit Price": round(price, 2),
            "PnL": round(pnl, 2)
        })
        position = None

trade_log = pd.DataFrame(trades)

# --- PnL Breakdown ---
if not trade_log.empty:
    trade_log['Entry Time'] = pd.to_datetime(trade_log['Entry Time'])
    trade_log['Exit Time'] = pd.to_datetime(trade_log['Exit Time'])
    trade_log['Month'] = trade_log['Exit Time'].dt.strftime('%b %Y')
    trade_log['Day'] = trade_log['Exit Time'].dt.strftime('%Y-%m-%d')

    monthly_pnl = trade_log.groupby('Month')['PnL'].sum().reset_index()
    daily_pnl = trade_log.groupby('Day')['PnL'].sum().reset_index()

    st.subheader("📅 Monthly PnL")
    st.dataframe(monthly_pnl)

    st.subheader("📆 Daily PnL")
    st.dataframe(daily_pnl)

# --- Trade Log Panel ---
st.subheader("📜 Trade Log")
st.dataframe(trade_log)

# --- CSV Export ---
st.download_button("📥 Export Trade Log", trade_log.to_csv(index=False), file_name=f"{symbol}_trade_log.csv")

# --- Plotly Chart ---
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DI_short'], mode='lines', name='DI Short', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DI_long'], mode='lines', name='DI Long', line=dict(color='green')))

for _, trade in trade_log.iterrows():
    fig.add_trace(go.Scatter(
        x=[trade['Entry Time']], y=[trade['Entry Price']],
        mode='markers', name='Buy', marker=dict(color='lime', size=10, symbol='triangle-up')
    ))
    fig.add_trace(go.Scatter(
        x=[trade['Exit Time']], y=[trade['Exit Price']],
        mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')
    ))

fig.update_layout(
    title=f"{symbol} Disparity Index Strategy",
    xaxis_title="Time (IST)",
    yaxis_title="DI Value (%)",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)
