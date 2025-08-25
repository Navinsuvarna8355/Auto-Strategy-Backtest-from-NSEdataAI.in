# app.py
# Streamlit aur anya zaroori libraries ko import karein
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- App ka UI aur setup ---
st.set_page_config(page_title="Share Market Backtest Tool", layout="wide")
st.title("💰 Automatic Share Market Backtest Tool")
st.write("Is tool mein, aap kisi bhi stock ka pichle 3 saal ka data download kar sakte hain aur Moving Average Crossover strategy ka upyog karke backtest kar sakte hain.")

# User se stock symbol input lene ke liye
stock_symbol = st.text_input("Stock Symbol (NSE ke liye '.NS' jod dein, jaise: RELIANCE.NS)", "RELIANCE.NS")

# Backtest shuru karne ke liye button
run_button = st.button("Backtest Chalao")

# --- Functions for Data and Backtest Logic ---

@st.cache_data
def get_historical_data(symbol, start_date, end_date):
    """
    Yahoo Finance se historical stock data download karta hai.
    Note: NSE se seedhe data download karna thoda mushkil hai, isliye
    main yaha yfinance library ka upyog kar raha hu jo reliable data deta hai.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Data download karte samay error hua: {e}. Kripya sahi symbol check karein.")
        return None

def run_backtest(df):
    """
    Moving Average Crossover strategy ke basis par backtest chalaata hai.
    Short-term MA > Long-term MA = Buy signal.
    Short-term MA < Long-term MA = Sell signal.
    """
    # Moving averages ki calculation
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Buy aur Sell signals generate karna
    df['Signal'] = 0.0 # 0 = koi signal nahi
    df['Signal'][20:] = df['SMA_20'][20:] > df['SMA_50'][20:]
    df['Positions'] = df['Signal'].diff()
    
    initial_capital = 100000 # Shuruaati nivesh
    positions = 0
    portfolio_value = initial_capital
    trade_log = []

    # Daily data par loop karke trades simulate karna
    for i in range(len(df)):
        current_date = df.index[i].strftime('%Y-%m-%d')
        
        # Buy signal - jab SMA_20, SMA_50 se upar jaata hai
        if df['Positions'].iloc[i] == 1.0:
            if positions == 0:
                buy_price = df['Close'].iloc[i]
                shares = initial_capital / buy_price # Saare paiso se shares kharidna
                positions = shares
                st.write(f"💼 **Buy Signal:** {current_date} par {shares:.2f} shares kharidein @ {buy_price:.2f}")
                
        # Sell signal - jab SMA_20, SMA_50 se niche jaata hai
        elif df['Positions'].iloc[i] == -1.0:
            if positions > 0:
                sell_price = df['Close'].iloc[i]
                profit_loss = (sell_price - buy_price) * positions
                portfolio_value += profit_loss
                
                trade_log.append({
                    'buy_date': buy_date,
                    'buy_price': buy_price,
                    'sell_date': current_date,
                    'sell_price': sell_price,
                    'profit_loss': profit_loss
                })
                
                positions = 0 # Saare shares bech diye
                st.write(f"🛑 **Sell Signal:** {current_date} par shares bechein @ {sell_price:.2f}. P/L: ₹{profit_loss:.2f}")
    
    # Aakhiri portfolio value calculate karna
    if positions > 0:
        final_value = df['Close'].iloc[-1] * positions
        portfolio_value += (final_value - (initial_capital - initial_capital))
    
    total_return = (portfolio_value - initial_capital) / initial_capital * 100
    
    return total_return, trade_log

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
                st.success("Data download safal raha!")
                
                # Candlestick chart dikhane ke liye
                st.subheader("Price Chart")
                st.line_chart(data[['Close', 'SMA_20', 'SMA_50']])
                
                st.subheader("Backtest Results")
                total_return, trade_log = run_backtest(data)
                
                # Results display karna
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Return", f"{total_return:.2f}%")
                col2.metric("Total Trades", len(trade_log))
                
                if trade_log:
                    trades_df = pd.DataFrame(trade_log)
                    winning_trades = trades_df[trades_df['profit_loss'] > 0]
                    col3.metric("Winning Trades", len(winning_trades))
                
                st.subheader("Trade History")
                if trade_log:
                    st.dataframe(pd.DataFrame(trade_log))
                else:
                    st.write("Is strategy ke liye koi trade nahi mila.")
            else:
                st.error("Diye gaye symbol ke liye data nahi mil paya. Kripya symbol check karein ya thodi der baad koshish karein.")

