import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_ohlc(symbol):
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(60)][::-1]
    base = 19500 if symbol == "NIFTY" else 44500
    prices = np.linspace(base, base + 100, 60) + np.random.normal(0, 10, 60)
    df = pd.DataFrame({"timestamp": timestamps, "close": prices})
    return df

