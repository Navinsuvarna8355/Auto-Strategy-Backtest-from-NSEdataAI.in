# app.py
# Import necessary libraries for Streamlit, data handling, and finance
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# --- App UI and Setup ---
st.set_page_config(page_title="Share Market Backtest Tool", layout="wide")
st.title("💰 Automatic Share Market Backtest Tool")
st.write("In this tool, you can download 3 years of historical data for any stock or index, like Nifty 50, and backtest it using the 'Disparity Index' strategy.")

# User inputs for stock symbol and strategy parameters
col_input, col_params = st.columns(2)
with col_input:
    # Example symbols for Nifty and Bank Nifty
    stock_symbol = st.text_input("Stock/Index Symbol (Append '.NS' for NSE stocks, use '^NSEI' for Nifty 50, and '^NSEBANK' for Bank Nifty)", "RELIANCE.NS")

with col_params:
    st.subheader("Disparity Index Parameters")
    length = st.number_input("Length (L)", min_value=1, value=29)
    short_period = st.number_input("Short Period", min_value=1, value=27)
    long_period = st.number_input("Long Period", min_value=1, value=81)

# Button to start the backtest
run_button = st.button("Run Backtest")

# --- Functions for Data and Backtest Logic ---

@st.cache_data
def get_historical_data(symbol, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred while downloading data: {e}. Please check the symbol.")
        return None

def calculate_disparity_index(df, length, short_period, long_period):
    """
    Calculates the Disparity Index and its EMAs based on the Pine Script formula.
    """
    # Calculate EMA, which will generate NaN values at the beginning
    df['EMA_Length'] = df['Close'].ewm(span=length, adjust=False).mean()
    
    # Calculate DI
    df['DI'] = ((df['Close'] - df['EMA_Length']) / df['EMA_Length']) * 100
    
    # Calculate DI's EMAs
    df['hsp_short'] = df['DI'].ewm(span=short_period, adjust=False).mean()
    df['hsp_long'] = df['DI'].ewm(span=long_period, adjust=False).mean()
    
    # After all calculations, drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def run_backtest(df, short_period, long_period):
    """
    Runs the backtest based on the Disparity Index strategy.
    hsp_short > hsp_long = Buy signal.
    hsp_short < hsp_long = Sell signal.
    """
    if short_period >= long_period:
        st.error("Short Period must be smaller than Long Period.")
        return None, None, None, None, None, None, None

    # Generate Buy and Sell signals
    df['Signal'] = np.where(df['hsp_short'] > df['hsp_long'], 1.0, 0.0)
    df['Positions'] = df['Signal'].diff()
    
    initial_capital = 100000 # Starting investment
    positions = 0
    portfolio_value = initial_capital
    trade_log = []
    
    # Track portfolio values over time for calculating drawdown
    portfolio_history = []
    
    buy_date = None
    buy_price = 0.0

    # Loop through the daily data to simulate trades
    for i in range(len(df)):
        current_date = df.index[i]
        
        # Buy signal - when hsp_short crosses above hsp_long
        if df['Positions'].iloc[i] == 1.0:
            if positions == 0:
                buy_price = df['Close'].iloc[i]
                buy_date = current_date
                shares = initial_capital / buy_price # Buy shares with all capital
                positions = shares
                
        # Sell signal - when hsp_short crosses below hsp_long
        elif df['Positions'].iloc[i] == -1.0:
            if positions > 0:
                sell_price = df['Close'].iloc[i]
                profit_loss = (sell_price - buy_price) * positions
                portfolio_value += profit_loss
                
                trade_log.append({
                    'buy_date': buy_date.strftime('%Y-%m-%d') if buy_date else 'N/A',
                    'buy_price': buy_price,
                    'sell_date': current_date.strftime('%Y-%m-%d'),
                    'sell_price': sell_price,
                    'profit_loss': profit_loss
                })
                
                positions = 0 # Sell all shares

        # Update portfolio value for the current day
        if positions > 0:
            current_value = df['Close'].iloc[i] * positions
        else:
            current_value = portfolio_value
        
        portfolio_history.append(current_value)

    # Calculate final portfolio value
    if positions > 0:
        final_value = df['Close'].iloc[-1] * positions
        profit_loss = (final_value - initial_capital)
        portfolio_value = initial_capital + profit_loss
    
    total_return = (portfolio_value - initial_capital) / initial_capital * 100
    
    # Calculate Max Drawdown
    portfolio_series = pd.Series(portfolio_history, index=df.index)
    peak = portfolio_series.cummax()
    drawdown = (portfolio_series - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

    # Calculate other trade metrics
    winning_trades = len([t for t in trade_log if t['profit_loss'] > 0])
    losing_trades = len([t for t in trade_log if t['profit_loss'] < 0])
    
    return total_return, trade_log, initial_capital, portfolio_value, max_drawdown, winning_trades, losing_trades

# --- Main App Logic ---
if run_button:
    if not stock_symbol:
        st.warning("Please enter a stock symbol.")
    else:
        with st.spinner("Downloading data and running backtest... Please wait."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3 * 365) # 3 years of data
            
            data = get_historical_data(stock_symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                # Calculate the Disparity Index after data download
                data = calculate_disparity_index(data, length, short_period, long_period)
                
                if not data.empty:
                    st.success("Data download successful!")
                    
                    # Display the price chart and Disparity Index
                    st.subheader("Price Chart and Disparity Index")
                    st.line_chart(data[['Close']])
                    st.line_chart(data[['hsp_short', 'hsp_long']])
                    
                    st.subheader("Backtest Results")
                    total_return, trade_log, initial_capital, final_portfolio_value, max_drawdown, winning_trades, losing_trades = run_backtest(data, short_period, long_period)
                    
                    if total_return is not None:
                        # Display the results
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Initial Capital", f"₹{initial_capital:.2f}")
                        col2.metric("Final Portfolio Value", f"₹{final_portfolio_value:.2f}")
                        col3.metric("Total Return", f"{total_return:.2f}%")
                        col4.metric("Total Trades", len(trade_log))

                        col5, col6, col7 = st.columns(3)
                        col5.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                        col6.metric("Winning Trades", winning_trades)
                        col7.metric("Losing Trades", losing_trades)

                        if trade_log:
                            st.subheader("Trade History")
                            trades_df = pd.DataFrame(trade_log)
                            st.dataframe(trades_df)
                        else:
                            st.write("No trades were found for this strategy.")
                else:
                    st.error("Could not find data for the given symbol. Please check the symbol or try again later.")
            else:
                st.error("Could not find data for the given symbol. Please check the symbol or try again later.")
