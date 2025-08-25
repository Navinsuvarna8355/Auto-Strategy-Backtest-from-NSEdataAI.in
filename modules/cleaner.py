import pandas as pd
import zipfile
import os

def extract_and_clean(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/bhavcopy/")
        csv_file = zip_path.replace(".zip", "")
        df = pd.read_csv(csv_file)
        df = df[df['SERIES'] == 'EQ']
        df = df[['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TIMESTAMP']]
        df.columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'date']
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
        return df
    except Exception as e:
        print(f"Clean error: {e}")
        return pd.DataFrame()

