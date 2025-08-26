import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="BTC Disparity Dashboard", layout="wide")
st.title("📊 BTC Strategy Dashboard")

# Sidebar: Manual strategy input
st.sidebar.header("⚙️ Strategy Settings")
param_input = st.sidebar.text_input(
    "Type settings like: MA=14, Short=5, Long=20",
    value="MA=14, Short=5, Long=20"
)

# Parse input safely
try:
    parts = param_input.replace(" ", "").split(",")
    param_dict = {k.split("=")[0].lower(): int(k.split("=")[1]) for k in parts}
    ma_length = param_dict.get("ma", 14)
    short_prd = param_dict.get("short", 5)
    long_prd = param_dict.get("long", 20)
except Exception as e:
    st.error(f"❌ Invalid format: {e}")
    st.stop()

# Display parsed values
st.subheader("🔍 Parsed Strategy Settings")
col1, col2, col3 = st.columns(3)
col1.metric("Disparity MA Length", ma_length)
col2.metric("Short Period", short_prd)
col3.metric("Long Period", long_prd)

# Load BTC data
@st.cache_data
def load_data():
    df = pd.read_csv("btc_data.csv")  # Must contain 'Date' and 'Close'
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Calculate Disparity Index
df['MA'] = df['Close'].rolling(window=ma_length).mean()
df['Disparity'] = (df['Close'] - df['MA']) / df['MA'] * 100

# Prepare chart data
chart_df = df[['Date', 'Close', 'Disparity']].dropna().set_index('Date')

# Plot chart using Streamlit native line_chart
st.subheader("📈 Disparity Index Chart")
st.line_chart(chart_df)

# Strategy signal
st.markdown("---")
if st.button("Run Strategy"):
    signal = "BUY" if short_prd > long_prd else "SELL"
    st.success(f"Signal: **{signal}** | MA={ma_length}, Short={short_prd}, Long={long_prd}")
