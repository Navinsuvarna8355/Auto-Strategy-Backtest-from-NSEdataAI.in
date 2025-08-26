import pandas as pd
import numpy as np

def apply_disparity_index(df, period):
    ma = df['close'].rolling(period).mean()
    df['DI'] = ((df['close'] - ma) / ma) * 100
    df['DI_short'] = df['DI']
    df['DI_long'] = df['DI'].rolling(5).mean()
    return df

def generate_signals(df, threshold):
    df['Signal'] = np.where(df['DI_short'] > threshold, 'Buy',
                     np.where(df['DI_short'] < -threshold, 'Sell', 'Hold'))
    return df

def simulate_trades(df, symbol):
    trades = []
    position = None
    entry_price = 0

    for i in range(1, len(df)):
        signal = df.loc[i, 'Signal']
        price = df.loc[i, 'close']
        time = df.loc[i, 'timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        if signal == 'Buy' and position is None:
            position = 'Long'
            entry_price = price
            entry_time = time
        elif signal == 'Sell' and position == 'Long':
            pnl = price - entry_price
            trades.append({
                "Symbol": symbol,
                "Entry Time": entry_time,
                "Entry Price": round(entry_price, 2),
                "Exit Time": time,
                "Exit Price": round(price, 2),
                "PnL": round(pnl, 2)
            })
            position = None

    return pd.DataFrame(trades)
