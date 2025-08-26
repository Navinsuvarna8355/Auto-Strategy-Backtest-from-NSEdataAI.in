import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- Strategy Logic ---
def calculate_disparity_index(df, length, short_prd, long_prd):
    ma = df['close'].rolling(length).mean()
    di = ((df['close'] - ma) / ma) * 100
    df['DI'] = di
    df['hsp_short'] = di.rolling(short_prd).mean()
    df['hsp_long'] = di.rolling(long_prd).mean()
    return df

def generate_trading_signals(disparity_results: pd.DataFrame):
    signals = pd.DataFrame(index=disparity_results.index)
    signals['signal'] = 0.0
    signals.loc[disparity_results['hsp_short'] > disparity_results['hsp_long'], 'signal'] = 1.0
    signals.loc[disparity_results['hsp_short'] < disparity_results['hsp_long'], 'signal'] = -1.0
    signals['entry'] = signals['signal'].diff()
    entry_signals = signals.loc[signals['entry'] != 0].copy()
    latest_signal = signals.iloc[-1]
    return entry_signals, latest_signal

# --- Streamlit UI ---
st.set_page_config(page_title="Disparity Index Strategy", layout="wide")
st.title("📊 Disparity Index Crossover Strategy")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    length = st.slider("Disparity MA Length", 5, 50, 14)
    short_prd = st.slider("Short Period", 2, 20, 5)
    long_prd = st.slider("Long Period", 5, 50, 20)

    st.header("📥 Data Input Mode")
    input_mode = st.radio("Choose input mode:", ["Auto-generate", "Manual"])

# --- Data Input ---
if input_mode == "Manual":
    st.sidebar.write("Enter comma-separated close prices:")
    manual_input = st.sidebar.text_area("Close Prices", "17500,17510,17520,17515,17530")
    try:
        close_prices = [float(p.strip()) for p in manual_input.split(",") if p.strip()]
        timestamps = pd.date_range(end=datetime.now(pytz.timezone('Asia/Kolkata')), periods=len(close_prices), freq='5min')
        df = pd.DataFrame({'close': close_prices}, index=timestamps)
    except Exception as e:
        st.error(f"Invalid input: {e}")
        st.stop()
else:
    np.random.seed(42)
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    dates = pd.date_range(end=now, periods=100, freq='5min')
    prices = np.cumsum(np.random.randn(len(dates))) + 17500
    df = pd.DataFrame({'close': prices}, index=dates)

# --- Strategy Execution ---
df = calculate_disparity_index(df, length, short_prd, long_prd)
entry_signals, latest_signal = generate_trading_signals(df)

# --- Display ---
st.subheader("📈 Latest Signal")
signal_text = "BUY" if latest_signal['signal'] == 1.0 else "SELL" if latest_signal['signal'] == -1.0 else "HOLD"
st.metric("Signal", signal_text)

st.subheader("📋 Entry Signals")
st.dataframe(entry_signals.tail(10))

st.subheader("📉 Disparity Index Chart")
st.line_chart(df[['DI', 'hsp_short', 'hsp_long']])

# --- CSV Export ---
csv = entry_signals.to_csv().encode('utf-8')
st.download_button("📥 Download Entry Signals CSV", data=csv, file_name="entry_signals.csv", mime="text/csv")
