# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pytz

# --- Streamlit Page Config ---
st.set_page_config(page_title="DI Backtest - Nifty & BankNifty", layout="wide")
st.title("📊 Disparity Index Backtest (Length=6, Short=9, Long=20)")
st.caption("Data: Last 3 years | Capital: ₹100,000 | Timestamps in IST")

# --- Symbol Map ---
symbols = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}

IST = pytz.timezone('Asia/Kolkata')

# --- Data Fetch ---
@st.cache_data
def fetch_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=3*365)
    df = yf.download(symbol, start=start, end=end)

    # Flatten MultiIndex if present (yfinance sometimes returns multi-level OHLC)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.dropna(inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert(IST)
    return df

# --- DI Calculation ---
def calculate_DI(df, length, short, long):
    close_series = df['Close'].astype(float)
    ema_series = close_series.ewm(span=length, adjust=False).mean()
    df['EMA'] = ema_series

    df['DI'] = ((close_series - ema_series) / ema_series) * 100
    df['DI_short'] = df['DI'].ewm(span=short, adjust=False).mean()
    df['DI_long'] = df['DI'].ewm(span=long, adjust=False).mean()

    df.dropna(inplace=True)
    return df

# --- Backtest Logic ---
def run_backtest(df):
    df['Signal'] = np.where(df['DI_short'] > df['DI_long'], 1.0, 0.0)
    df['Position'] = df['Signal'].diff()

    capital = 100000
    position = 0
    entry_price = None
    trade_log = []
    portfolio = capital
    history = []

    for i in range(len(df)):
        current_date = df.index[i]
        price = df['Close'].iloc[i]

        # Entry
        if df['Position'].iloc[i] == 1.0 and position == 0:
            entry_price = price
            entry_date = current_date
            shares = capital / price
            position = shares

        # Exit
        elif df['Position'].iloc[i] == -1.0 and position > 0:
            exit_price = price
            exit_date = current_date
            pnl = (exit_price - entry_price) * position
            portfolio += pnl
            trade_log.append({
                "Buy Date": entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                "Buy Price": entry_price,
                "Sell Date": exit_date.strftime('%Y-%m-%d %H:%M:%S'),
                "Sell Price": exit_price,
                "PnL": pnl
            })
            position = 0

        current_value = price * position if position > 0 else portfolio
        history.append(current_value)

    final_value = price * position if position > 0 else portfolio
    total_return = (final_value - capital) / capital * 100
    drawdown = pd.Series(history).cummax() - pd.Series(history)
    max_dd = (drawdown.max() / pd.Series(history).cummax().max()) * 100

    wins = len([t for t in trade_log if t['PnL'] > 0])
    losses = len([t for t in trade_log if t['PnL'] < 0])

    return total_return, final_value, max_dd, wins, losses, pd.DataFrame(trade_log)

# --- Main Loop for Symbols ---
for label, symbol in symbols.items():
    st.subheader(f"📈 {label} ({symbol})")

    df = fetch_data(symbol)
    df = calculate_DI(df, length=6, short=9, long=20)

    # Price chart
    st.line_chart(df[['Close']])

    # DI chart
    st.line_chart(df[['DI_short', 'DI_long']])

    # Backtest results
    total_return, final_value, max_dd, wins, losses, log_df = run_backtest(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Capital", "₹100000")
    col2.metric("Final Value", f"₹{final_value:,.2f}")
    col3.metric("Total Return", f"{total_return:.2f}%")
    col4.metric("Max Drawdown", f"{max_dd:.2f}%")

    col5, col6 = st.columns(2)
    col5.metric("Winning Trades", wins)
    col6.metric("Losing Trades", losses)

    st.subheader("📜 Trade Log")
    st.dataframe(log_df)

    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download {label} Trade Log",
        data=csv,
        file_name=f"{label}_trade_log.csv",
        mime="text/csv"
    )
