import pandas as pd
import pytz
import streamlit as st

def convert_to_IST(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
    return df

def export_csv(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Trade Log", data=csv, file_name=filename, mime='text/csv')

