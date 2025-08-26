import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from strategy import apply_disparity_index, generate_signals, simulate_trades
from utils import breakdown_pnl, export_csv
from backtest import run_backtest  # ✅ NEW IMPORT

st.set_page_config(layout="wide")
st.title("📈 Disparity Index Dashboard")

# --- Strategy Inputs ---
st.sidebar.header("Disparity Index Settings")
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
length = st.sidebar.slider("Length (MA)", 1, 50, 3)
short_period = st.sidebar.slider("Short Period (DI Short)", 1, 20, 4)
long_period = st.sidebar.slider("Long Period (DI Long)", 5, 50, 20)
threshold = st.sidebar.slider("Threshold (%)", 0.5, 5.0, 2.0)

# --- Live OHLC Simulation ---
from fetcher import get_ohlc
df = get_ohlc(symbol)
df = apply_disparity_index(df, length, short_period, long_period)
df = generate_signals(df, threshold)
trade_log = simulate_trades(df, symbol)

monthly_pnl, daily_pnl = breakdown_pnl(trade_log)

st.subheader("📅 Monthly PnL")
st.dataframe(monthly_pnl)
st.subheader("📆 Daily PnL")
st.dataframe(daily_pnl)
st.subheader("📜 Trade Log")
st.dataframe(trade_log)
export_csv(trade_log, f"{symbol}_trade_log.csv")

# --- Live Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DI_short'], mode='lines', name='DI Short', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['DI_long'], mode='lines', name='DI Long', line=dict(color='green')))
for _, trade in trade_log.iterrows():
    fig.add_trace(go.Scatter(x=[trade['Entry Time']], y=[trade['Entry Price']],
                             mode='markers', name='Buy', marker=dict(color='lime', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=[trade['Exit Time']], y=[trade['Exit Price']],
                             mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
fig.update_layout(title=f"{symbol} Live Strategy", xaxis_title="Time", yaxis_title="DI Value (%)", template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# --- ✅ Auto Backtest Section ---
uploaded_file = st.sidebar.file_uploader("📁 Upload OHLC CSV for Backtest", type=["csv"])
if uploaded_file:
    st.subheader("🧪 Backtest Results")
    df_bt = pd.read_csv(uploaded_file)
    df_bt, trade_log_bt, monthly_pnl_bt, daily_pnl_bt = run_backtest(
        df_bt, symbol, length, short_period, long_period, threshold
    )

    st.markdown("**📅 Monthly PnL**")
    st.dataframe(monthly_pnl_bt)
    st.markdown("**📆 Daily PnL**")
    st.dataframe(daily_pnl_bt)
    st.markdown("**📜 Trade Log**")
    st.dataframe(trade_log_bt)
    export_csv(trade_log_bt, f"{symbol}_backtest_log.csv")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=df_bt['timestamp'], y=df_bt['DI_short'], mode='lines', name='DI Short', line=dict(color='blue')))
    fig_bt.add_trace(go.Scatter(x=df_bt['timestamp'], y=df_bt['DI_long'], mode='lines', name='DI Long', line=dict(color='green')))
    for _, trade in trade_log_bt.iterrows():
        fig_bt.add_trace(go.Scatter(x=[trade['Entry Time']], y=[trade['Entry Price']],
                                    mode='markers', name='Buy', marker=dict(color='lime', size=10, symbol='triangle-up')))
        fig_bt.add_trace(go.Scatter(x=[trade['Exit Time']], y=[trade['Exit Price']],
                                    mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
    fig_bt.update_layout(title="Backtest Chart", xaxis_title="Time", yaxis_title="DI Value (%)", template="plotly_dark", height=500)
    st.plotly_chart(fig_bt, use_container_width=True)
