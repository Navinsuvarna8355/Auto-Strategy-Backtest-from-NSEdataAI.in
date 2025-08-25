# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# --- UI Setup ---
st.set_page_config(page_title="Nifty & BankNifty Backtest", layout="wide")
st.title("📊 Disparity Index Backtest: Nifty & BankNifty")
st.caption("Backtest using DI settings: Length=6, Short=9, Long=20 | 3-Year Data")

symbols = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}

# --- Functions ---
@st.cache_data
def fetch_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=3*365)
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    return df

def calculate_DI(df, length, short, long):
    df['EMA'] = df['Close'].ewm(span=length, adjust=False).mean()
    df.dropna(inplace=True)
    df['DI'] = ((df['Close'] - df['EMA']) / df['EMA']) * 100
    df['DI_short'] = df['DI'].ewm(span=short, adjust=False).mean()
    df['DI_long'] = df['DI'].ewm(span=long, adjust=False).mean()
    df.dropna(inplace=True)
    return df

def run_backtest(df):
    df['Signal'] = np.where(df['DI_short'] > df['DI_long'], 1.0, 0.0)
    df['Position'] = df['Signal'].diff()

    capital = 100000
    position = 0
    trade_log = []
    portfolio = capital
    history = []

    for i in range(len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]

        if df['Position'].iloc[i] == 1.0 and position == 0:
            entry_price = price
            entry_date = date
            shares = capital / price
            position = shares

        elif df['Position'].iloc[i] == -1.0 and position > 0:
            exit_price = price
            exit_date = date
            pnl = (exit_price - entry_price) * position
            portfolio += pnl
            trade_log.append({
                "Buy Date": entry_date.strftime('%Y-%m-%d'),
                "Buy Price": entry_price,
                "Sell Date": exit_date.strftime('%Y-%m-%d'),
                "Sell Price": exit_price,
                "PnL": pnl
            })
            position = 0

        current_value = price * position if position > 0 else portfolio
        history.append(current_value)

    final_value = price * position if position > 0 else portfolio
    total_return = (final_value - capital) / capital * 100
    drawdown = pd.Series(history).cummax() - pd.Series(history)
    max_dd = drawdown.max() / pd.Series(history).cummax().max() * 100

    wins = len([t for t in trade_log if t['PnL'] > 0])
    losses = len([t for t in trade_log if t['PnL'] < 0])

    return total_return, final_value, max_dd, wins, losses, pd.DataFrame(trade_log)

# --- Main App ---
for label, symbol in symbols.items():
    st.subheader(f"📈 {label} ({symbol})")
    df = fetch_data(symbol)
    df = calculate_DI(df, length=6, short=9, long=20)
    st.line_chart(df[['Close']])
    st.line_chart(df[['DI_short', 'DI_long']])

    total_return, final_value, max_dd, wins, losses, log_df = run_backtest(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Capital", "₹100000")
    col2.metric("Final Value", f"₹{final_value:.2f}")
    col3.metric("Total Return", f"{total_return:.2f}%")
    col4.metric("Max Drawdown", f"{max_dd:.2f}%")

    col5, col6 = st.columns(2)
    col5.metric("Winning Trades", wins)
    col6.metric("Losing Trades", losses)

    st.subheader("📜 Trade Log")
    st.dataframe(log_df)

    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(f"Download {label} Trade Log", data=csv, file_name=f"{label}_trade_log.csv", mime="text/csv")
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
                    st.error("Could not find enough data for the given symbol to run the backtest. Please check the symbol or try again later.")
            else:
                st.error("Could not find data for the given symbol. Please check the symbol or try again later.")
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
                    st.error("Could not find enough data for the given symbol to run the backtest. Please check the symbol or try again later.")
            else:
                st.error("Could not find data for the given symbol. Please check the symbol or try again later.")
