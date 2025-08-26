import pandas as pd

def run_backtest(df):
    trades = []
    position = None
    entry_price = 0
    entry_time = None

    for i in range(1, len(df)):
        signal = df.iloc[i]['signal']
        price = df.iloc[i]['close']
        time = df.iloc[i]['timestamp']

        if signal == 1 and position is None:
            position = 'long'
            entry_price = price
            entry_time = time

        elif signal == -1 and position == 'long':
            pnl = price - entry_price
            trades.append({
                'Entry Time': entry_time,
                'Exit Time': time,
                'Entry Price': entry_price,
                'Exit Price': price,
                'P&L': round(pnl, 2)
            })
            position = None

    trade_log = pd.DataFrame(trades)
    trade_log['Entry Time'] = pd.to_datetime(trade_log['Entry Time'])
    trade_log['Exit Time'] = pd.to_datetime(trade_log['Exit Time'])

    trade_log['Date'] = trade_log['Exit Time'].dt.date
    trade_log['Month'] = trade_log['Exit Time'].dt.to_period('M')

    pnl_daily = trade_log.groupby('Date')['P&L'].sum().reset_index()
    pnl_monthly = trade_log.groupby('Month')['P&L'].sum().reset_index()

    return trade_log, pnl_daily, pnl_monthly

