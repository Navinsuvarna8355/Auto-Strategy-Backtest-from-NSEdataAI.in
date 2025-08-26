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

def generate_trading_signals(disparity_results: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=disparity_results.index)
    signals['signal'] = 0.0
    signals.loc[disparity_results['hsp_short'] > disparity_results['hsp_long'], 'signal'] = 1.0
    signals.loc[disparity_results['hsp_short'] < disparity_results['hsp_long'], 'signal'] = -1.0
    signals['entry'] = signals['signal'].diff()
    entry_signals = signals.loc[signals['entry'] != 0].copy()
    latest_signal = signals.iloc[-1]
    return entry_signals, latest_signal

# --- Sample OHLC Data Generator ---
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    prices = np.cumsum(np.random.randn(len(dates))) + 17500
    df = pd.DataFrame({'close': prices}, index=dates)
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Disparity Index Strategy", layout="wide")
st.title("📊 Disparity Index Crossover Strategy")

with st.sidebar:
    st.header("⚙️ Strategy Settings")
    length = st.slider("Disparity MA Length", 5, 50, 14)
    short_prd = st.slider("Short Period", 2, 20, 5)
    long_prd = st.slider("Long Period", 5, 50, 20)

# --- Data & Strategy Execution ---
df = generate_sample_data()
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

# --- Optional CSV Export ---
csv = entry_signals.to_csv().encode('utf-8')
st.download_button("📥 Download Entry Signals CSV", data=csv, file_name="entry_signals.csv", mime="text/csv")

