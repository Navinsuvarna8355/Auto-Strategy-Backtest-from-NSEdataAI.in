from strategy import apply_disparity_index, generate_signals, simulate_trades
from utils import breakdown_pnl
import pandas as pd

def run_backtest(df, symbol, length, short_period, long_period, threshold):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = apply_disparity_index(df, length, short_period, long_period)
    df = generate_signals(df, threshold)
    trade_log = simulate_trades(df, symbol)
    monthly_pnl, daily_pnl = breakdown_pnl(trade_log)
    return df, trade_log, monthly_pnl, daily_pnl
