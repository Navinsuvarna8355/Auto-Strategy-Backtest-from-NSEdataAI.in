import streamlit as st
from strategy import generate_signals
from backtest import run_backtest
from utils import convert_to_IST

st.title("📈 Disparity Index Backtest Tool for NIFTY/BANKNIFTY")

length = st.slider("Length", min_value=1, max_value=50, value=29)
short_period = st.slider("Short Period", min_value=1, max_value=50, value=27)
long_period = st.slider("Long Period", min_value=1, max_value=100, value=81)

uploaded_file = st.file_uploader("Upload 5-min OHLC CSV", type=["csv"])
if uploaded_file:
    df = convert_to_IST(uploaded_file)
    df = generate_signals(df, length, short_period, long_period)
    trade_log, pnl_daily, pnl_monthly = run_backtest(df)

    st.subheader("📋 Trade Log")
    st.dataframe(trade_log)

    st.subheader("📆 Daily P&L")
    st.dataframe(pnl_daily)

    st.subheader("🗓️ Monthly P&L")
    st.dataframe(pnl_monthly)
import streamlit as st
import importlib
from datetime import datetime

# ------------------------
# Env Check – Non‑Blocking
# ------------------------
REQUIRED_MODULES = ["pandas", "numpy", "plotly"]

missing = []
for mod in REQUIRED_MODULES:
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)

if missing:
    st.error(f"⚠ Missing modules: {', '.join(missing)} (app will still load)")

# ------------------------
# Utility – IST Timestamp
# ------------------------
def ist_now(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.utcnow().astimezone(
        tz=datetime.now().astimezone().tzinfo
    ).strftime(fmt)

# ------------------------
# Main App
# ------------------------
def main():
    st.set_page_config(page_title="Disparity Index Dashboard", layout="wide")
    st.title("📊 Dual‑Symbol Auto Paper Trading – DI Safe Mode")

    # Sidebar Config
    st.sidebar.header("Settings")
    disparity_period = st.sidebar.number_input("DI Period", 5, 50, 14)
    disparity_threshold = st.sidebar.number_input("DI Threshold (%)", 0.1, 5.0, 1.5)

    if st.sidebar.button("🚀 Start Paper Trading"):
        run_paper_trading(disparity_period, disparity_threshold)
    else:
        st.info("⚡ Ready – Press **Start** to fetch live data & trade.")

# ------------------------
# Core Loop
# ------------------------
def run_paper_trading(period, threshold):
    st.success(f"Paper trading started at {ist_now()} with DI={period}, Threshold={threshold}")
    # 🛑 Heavy data fetch & logic goes here – run in threads if needed
    # Example placeholder:
    st.write("Fetching OHLC & computing Disparity Index…")
    # add_live_feed()
    # calc_signals()
    # execute_paper_trade()

# ------------------------
# Safe Entry Point
# ------------------------
if __name__ == "__main__":
    main()
