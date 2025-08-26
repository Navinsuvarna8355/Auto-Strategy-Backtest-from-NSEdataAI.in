import streamlit as st
import pandas as pd
from strategy import generate_signals
from backtest import run_backtest
from utils import convert_to_IST, export_csv

st.set_page_config(page_title="Disparity Index Backtest", layout="wide")
st.title("📈 Disparity Index Backtest Tool for NIFTY / BANKNIFTY")

# Strategy Inputs
length = st.slider("Length", 1, 50, 29)
short_period = st.slider("Short Period", 1, 50, 27)
long_period = st.slider("Long Period", 1, 100, 81)

# File Upload
uploaded_file = st.file_uploader("Upload 5-min OHLC CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = convert_to_IST(df)
    df = generate_signals(df, length, short_period, long_period)
    
    trade_log, pnl_daily, pnl_monthly = run_backtest(df)

    st.subheader("📋 Trade Log")
    st.dataframe(trade_log)
    export_csv(trade_log, "Trade_Log.csv")

    st.subheader("📆 Daily P&L")
    st.dataframe(pnl_daily)

    st.subheader("🗓️ Monthly P&L")
    st.dataframe(pnl_monthly)
