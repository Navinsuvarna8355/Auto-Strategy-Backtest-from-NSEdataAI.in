import requests, pandas as pd, numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from scipy.stats import norm

# 🔹 Step 1: Fetch Option Chain Data
def fetch_option_chain(symbol="NIFTY", expiry="28-Aug-2025"):
    url = f"https://www.nsedataai.in/api/option-chain?symbol={symbol}&expiry={expiry}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['records'])
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # IST
        return df
    except Exception as e:
        print(f"⚠️ Fetch failed: {e}")
        return pd.DataFrame()

# 🔹 Step 2: Calculate Greeks Locally
def calculate_greeks(df, r=0.06):
    greeks = []
    for _, row in df.iterrows():
        try:
            S = row['underlyingValue']
            K = row['strikePrice']
            T = row['daysToExpiry'] / 365
            IV = row['impliedVolatility'] / 100
            option_type = row['optionType']
            
            d1 = (np.log(S/K) + (r + IV**2/2)*T) / (IV*np.sqrt(T))
            d2 = d1 - IV*np.sqrt(T)
            
            delta = norm.cdf(d1) if option_type == 'CE' else -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * IV * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            greeks.append({
                "strike": K, "type": option_type,
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "vega": round(vega, 4)
            })
        except:
            continue
    return pd.DataFrame(greeks)

# 🔹 Step 3: Strategy Logic (Disparity Index + EMA)
def disparity_index(close, ema):
    return ((close - ema) / ema) * 100

def run_strategy(df):
    df['EMA'] = df['close'].ewm(span=21).mean()
    df['Disparity'] = disparity_index(df['close'], df['EMA'])
    df['Signal'] = df['Disparity'].apply(lambda x: 'BUY' if x < -2 else ('SELL' if x > 2 else 'HOLD'))
    df['PnL'] = df['Signal'].map({'BUY': 100, 'SELL': -100, 'HOLD': 0})  # Placeholder logic
    return df[['timestamp', 'close', 'EMA', 'Disparity', 'Signal', 'PnL']]

# 🔹 Step 4: Trade Log Export
def export_trade_log(df, filename="trade_log.csv"):
    df.to_csv(filename, index=False)
    print(f"✅ Trade log saved: {filename}")

# 🔹 Step 5: Full Pipeline Trigger
def run_full_backtest(symbol="NIFTY", expiry="28-Aug-2025"):
    raw_df = fetch_option_chain(symbol, expiry)
    if raw_df.empty:
        print("❌ No data to process.")
        return
    
    greeks_df = calculate_greeks(raw_df)
    strategy_df = run_strategy(raw_df)
    merged = pd.concat([strategy_df.reset_index(drop=True), greeks_df.reset_index(drop=True)], axis=1)
    
    export_trade_log(merged, f"{symbol}_backtest_log.csv")
    return merged

