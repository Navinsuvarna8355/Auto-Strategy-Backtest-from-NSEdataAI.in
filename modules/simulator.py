import pandas as pd

def simulate_trades(df):
    trades = []
    position = None
    for _, row in df.iterrows():
        if row['Signal'] == 'BUY' and not position:
            position = row['close']
            entry_date = row['date']
        elif row['Signal'] == 'SELL' and position:
            pnl = row['close'] - position
            trades.append({
                'entry_date': entry_date,
                'entry_price': position,
                'exit_date': row['date'],
                'exit_price': row['close'],
                'PnL': round(pnl, 2)
            })
            position = None
    return pd.DataFrame(trades)

