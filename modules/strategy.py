def apply_ma_strategy(df, short_window=5, long_window=20):
    df = df.sort_values('date')
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    df['Signal'] = df.apply(lambda row: 'BUY' if row['SMA_short'] > row['SMA_long'] else 'SELL', axis=1)
    return df

