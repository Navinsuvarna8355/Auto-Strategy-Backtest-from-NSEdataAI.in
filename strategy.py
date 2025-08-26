import pandas as pd

def disparity_index(close, length, period):
    ema_base = close.ewm(span=length, adjust=False).mean()
    disparity = ((close - ema_base) / ema_base) * 100
    return disparity.ewm(span=period, adjust=False).mean()

def generate_signals(df, length, short_period, long_period):
    df['DI_short'] = disparity_index(df['close'], length, short_period)
    df['DI_long'] = disparity_index(df['close'], length, long_period)
    
    df['signal'] = 0
    df.loc[df['DI_short'] > df['DI_long'], 'signal'] = 1
    df.loc[df['DI_short'] < df['DI_long'], 'signal'] = -1
    return df

