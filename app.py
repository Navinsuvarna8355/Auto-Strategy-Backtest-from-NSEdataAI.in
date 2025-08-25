# app.py
# Streamlit aur anya zaroori libraries ko import karein
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# --- App ka UI aur setup ---
st.set_page_config(page_title="Share Market Backtest Tool", layout="wide")
st.title("💰 Automatic Share Market Backtest Tool")
st.write("Is tool mein, aap kisi bhi stock ka pichle 3 saal ka data download kar sakte hain aur 'Disparity Index' strategy ka upyog karke backtest kar sakte hain.")

# User se stock symbol aur strategy parameters input lene ke liye
col_input, col_params = st.columns(2)
with col_input:
    stock_symbol = st.text_input("Stock Symbol (NSE ke liye '.NS' jod dein, jaise: RELIANCE.NS)", "RELIANCE.NS")

with col_params:
    st.subheader("Disparity Index Parameters")
    length = st.number_input("Length (L)", min_value=1, value=29)
    short_period = st.number_input("Short Period", min_value=1, value=27)
    long_period = st.number_input("Long Period", min_value=1, value=81)

# Backtest shuru karne ke liye button
run_button = st.button("Backtest Chalao")

# --- Functions for Data and Backtest Logic ---

@st.cache_data
def get_historical_data(symbol, start_date, end_date):
    """
    Yahoo Finance se historical stock data download karta hai.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Data download karte samay error hua: {e}. Kripya sahi symbol check karein.")
        return None

def calculate_disparity_index(df, length, short_period, long_period):
    """
    Pine Script formula ke hisaab se Disparity Index aur uske EMAs ki calculation karta hai.
    """
    # EMA ki calculation
    df['EMA_Length'] = df['Close'].ewm(span=length, adjust=False).mean()
    
    # DI calculation
    df['DI'] = ((df['Close'] - df['EMA_Length']) / df['EMA_Length']) * 100
    
    # DI ke EMA calculations
    df['hsp_short'] = df['DI'].ewm(span=short_period, adjust=False).mean()
    df['hsp_long'] = df['DI'].ewm(span=long_period, adjust=False).mean()

    # Drop any rows with NaN values after calculations
    df.dropna(inplace=True)
    
    return df

def run_backtest(df, short_period, long_period):
    """
    Disparity Index strategy ke basis par backtest chalaata hai.
    hsp_short > hsp_long = Buy signal.
    hsp_short < hsp_long = Sell signal.
    """
    if short_period >= long_period:
        st.error("Short Period, Long Period se chhota hona chahiye.")
        return None, None, None, None, None, None, None

    # Buy aur Sell signals generate karna
    df['Signal'] = np.where(df['hsp_short'] > df['hsp_long'], 1.0, 0.0)
    df['Positions'] = df['Signal'].diff()
    
    initial_capital = 100000 # Shuruaati nivesh
    positions = 0
    portfolio_value = initial_capital
    trade_log = []
    
    # Track portfolio values over time for calculating drawdown
    portfolio_history = []
    
    buy_date = None
    buy_price = 0.0

    # Daily data par loop karke trades simulate karna
    for i in range(len(df)):
        current_date = df.index[i]
        
        # Buy signal - jab hsp_short, hsp_long se upar jaata hai
        if df['Positions'].iloc[i] == 1.0:
            if positions == 0:
                buy_price = df['Close'].iloc[i]
                buy_date = current_date
                shares = initial_capital / buy_price # Saare paiso se shares kharidna
                positions = shares
                
        # Sell signal - jab hsp_short, hsp_long se niche jaata hai
        elif df['Positions'].iloc[i] == -1.0:
            if positions > 0:
                sell_price = df['Close'].iloc[i]
                profit_loss = (sell_price - buy_price) * positions
                portfolio_value += profit_loss
                
                trade_log.append({
                    'buy_date': buy_date.strftime('%Y-%m-%d'),
                    'buy_price': buy_price,
                    'sell_date': current_date.strftime('%Y-%m-%d'),
                    'sell_price': sell_price,
                    'profit_loss': profit_loss
                })
                
                positions = 0 # Saare shares bech diye

        # Update portfolio value for current day
        if positions > 0:
            current_value = df['Close'].iloc[i] * positions
        else:
            current_value = portfolio_value
        
        portfolio_history.append(current_value)

    # Aakhiri portfolio value calculate karna
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

# --- Main app logic ---
if run_button:
    if not stock_symbol:
        st.warning("Kripya stock symbol daalein.")
    else:
        with st.spinner("Data download aur backtesting chal raha hai... Kripya intezar karein."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3 * 365) # Pura 3 saal ka data
            
            data = get_historical_data(stock_symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                # Disparity Index ki calculation data download hone ke baad karein
                data = calculate_disparity_index(data, length, short_period, long_period)
                
                if not data.empty:
                    st.success("Data download safal raha!")
                    
                    # Candlestick chart dikhane ke liye
                    st.subheader("Price Chart aur Disparity Index")
                    st.line_chart(data[['Close']])
                    st.line_chart(data[['hsp_short', 'hsp_long']])
                    
                    st.subheader("Backtest Results")
                    total_return, trade_log, initial_capital, final_portfolio_value, max_drawdown, winning_trades, losing_trades = run_backtest(data, short_period, long_period)
                    
                    if total_return is not None:
                        # Results display karna
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
                            st.write("Is strategy ke liye koi trade nahi mila.")
                else:
                    st.error("Diye gaye symbol ke liye data nahi mil paya. Kripya symbol check karein ya thodi der baad koshish karein.")
            else:
                st.error("Diye gaye symbol ke liye data nahi mil paya. Kripya symbol check karein ya thodi der baad koshish karein.")
                positions = 0 # Saare shares bech diye

        # Update portfolio value for current day
        if positions > 0:
            current_value = df['Close'].iloc[i] * positions
        else:
            current_value = portfolio_value
        
        portfolio_history.append(current_value)

    # Aakhiri portfolio value calculate karna
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

# --- Main app logic ---
if run_button:
    if not stock_symbol:
        st.warning("Kripya stock symbol daalein.")
    else:
        with st.spinner("Data download aur backtesting chal raha hai... Kripya intezar karein."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3 * 365) # Pura 3 saal ka data
            
            data = get_historical_data(stock_symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                # Disparity Index ki calculation data download hone ke baad karein
                data = calculate_disparity_index(data, length, short_period, long_period)

                st.success("Data download safal raha!")
                
                # Candlestick chart dikhane ke liye
                st.subheader("Price Chart aur Disparity Index")
                # Disparity Index lines alag se plot karna behtar hai, kyunki scale alag hota hai
                st.line_chart(data[['Close']])
                st.line_chart(data[['hsp_short', 'hsp_long']])
                
                st.subheader("Backtest Results")
                total_return, trade_log, initial_capital, final_portfolio_value, max_drawdown, winning_trades, losing_trades = run_backtest(data, short_period, long_period)
                
                if total_return is not None:
                    # Results display karna
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
                        st.write("Is strategy ke liye koi trade nahi mila.")
            else:
                st.error("Diye gaye symbol ke liye data nahi mil paya. Kripya symbol check karein ya thodi der baad koshish karein.")
                    'sell_date': current_date,
                    'sell_price': sell_price,
                    'profit_loss': profit_loss
                })
                
                positions = 0 # Saare shares bech diye
                st.write(f"🛑 **Sell Signal:** {current_date} par shares bechein @ {sell_price:.2f}. P/L: ₹{profit_loss:.2f}")
    
    # Aakhiri portfolio value calculate karna
    if positions > 0:
        final_value = df['Close'].iloc[-1] * positions
        profit_loss = (final_value - initial_capital)
        portfolio_value = initial_capital + profit_loss
        
    total_return = (portfolio_value - initial_capital) / initial_capital * 100
    
    return total_return, trade_log, initial_capital, portfolio_value

# --- Main app logic ---
if run_button:
    if not stock_symbol:
        st.warning("Kripya stock symbol daalein.")
    else:
        with st.spinner("Data download aur backtesting chal raha hai... Kripya intezar karein."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3 * 365) # Pura 3 saal ka data
            
            data = get_historical_data(stock_symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                # Disparity Index ki calculation data download hone ke baad karein
                data = calculate_disparity_index(data, length, short_period, long_period)

                st.success("Data download safal raha!")
                
                # Candlestick chart dikhane ke liye
                st.subheader("Price Chart aur Disparity Index")
                # Disparity Index lines alag se plot karna behtar hai, kyunki scale alag hota hai
                st.line_chart(data[['Close']])
                st.line_chart(data[['hsp_short', 'hsp_long']])
                
                st.subheader("Backtest Results")
                total_return, trade_log, initial_capital, final_portfolio_value = run_backtest(data, short_period, long_period)
                
                if total_return is not None:
                    # Results display karna
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Initial Capital", f"₹{initial_capital:.2f}")
                    col2.metric("Final Portfolio Value", f"₹{final_portfolio_value:.2f}")
                    col3.metric("Total Return", f"{total_return:.2f}%")
                    col4.metric("Total Trades", len(trade_log))

                    if trade_log:
                        st.subheader("Trade History")
                        trades_df = pd.DataFrame(trade_log)
                        st.dataframe(trades_df)
                    else:
                        st.write("Is strategy ke liye koi trade nahi mila.")
            else:
                st.error("Diye gaye symbol ke liye data nahi mil paya. Kripya symbol check karein ya thodi der baad koshish karein.")
