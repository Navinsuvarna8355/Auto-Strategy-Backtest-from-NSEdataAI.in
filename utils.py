import pandas as pd
import streamlit as st
import pytz

def to_ist(dt):
    ist = pytz.timezone('Asia/Kolkata')
    return dt.astimezone(ist).strftime('%Y-%m-%d %H:%M:%S')

def breakdown_pnl(trade_log):
    if trade_log.empty:
        return pd.DataFrame(), pd.DataFrame()

    trade_log['Entry Time'] = pd.to_datetime(trade_log['Entry Time'])
    trade_log['Exit Time'] = pd.to_datetime(trade_log['Exit Time'])
    trade_log['Month'] = trade_log['Exit Time'].dt.strftime('%b %Y')
    trade_log['Day'] = trade_log['Exit Time'].dt.strftime('%Y-%m-%d')

    monthly_pnl = trade_log.groupby('Month')['PnL'].sum().reset_index()
    daily_pnl = trade_log.groupby('Day')['PnL'].sum().reset_index()
    return monthly_pnl, daily_pnl

def export_csv(df, filename):
    st.download_button("📥 Export Trade Log", df.to_csv(index=False), file_name=filename)
