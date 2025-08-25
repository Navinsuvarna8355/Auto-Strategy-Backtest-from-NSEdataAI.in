import os
import io
import time
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

IST = pytz.timezone("Asia/Kolkata")

# ----------------------------
# Utilities
# ----------------------------
def to_ist(df, tz_col="index"):
    if tz_col == "index":
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        return df.tz_convert(IST)
    else:
        if pd.api.types.is_datetime64_any_dtype(df[tz_col]):
            s = df[tz_col]
            if s.dt.tz is None:
                s = s.dt.tz_localize("UTC")
            df[tz_col] = s.dt.tz_convert(IST)
        return df

def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def compute_disparity_index(close, ma_len):
    base = ema(close, ma_len)
    di = (close / base) * 100.0
    return di, base

def max_period_for_interval(interval):
    # yfinance limits for intraday
    return {
        "5m": 60,     # days
        "15m": 60,
        "60m": 730,
        "1d": 1825,
    }.get(interval, 365)

def yf_interval(tf):
    return {"5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}[tf]

def fetch_yf(symbol, timeframe, lookback_days):
    interval = yf_interval(timeframe)
    maxd = max_period_for_interval(interval)
    period_days = min(lookback_days, maxd)
    period = f"{period_days}d"

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)  # ensure Title case: Open, High, Low, Close, Volume
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df = to_ist(df)
    df.index.name = "Datetime"
    return df

def parse_csv(file):
    # Expect columns: Datetime, Open, High, Low, Close, Volume
    df = pd.read_csv(file)
    # Flexible datetime parsing
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.set_index("Datetime")
    else:
        # Try first column as datetime
        first = df.columns[0]
        df[first] = pd.to_datetime(df[first], utc=True, errors="coerce")
        df = df.set_index(first)
    # Standardize columns
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for std in ["open", "high", "low", "close", "volume"]:
        if std in cols:
            rename_map[cols[std]] = std.title()
    df = df.rename(columns=rename_map)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = to_ist(df)
    df.index.name = "Datetime"
    df.dropna(inplace=True)
    return df

def next_open_prices(df):
    return df["Open"].shift(-1)

# ----------------------------
# Strategy & backtest
# ----------------------------
def generate_signals(df, di_len, sig_len, threshold):
    di, base = compute_disparity_index(df["Close"], di_len)
    sig = ema(di, sig_len)

    # Cross rules with threshold and trend filter using signal EMA
    cross_up = (di > threshold) & (di.shift(1) <= threshold)
    cross_dn = (di < threshold) & (di.shift(1) >= threshold)

    # Optional confirmation by DI vs its EMA
    long_on = cross_up & (di > sig)
    long_off = cross_dn | (di < sig)

    pos = pd.Series(0, index=df.index, dtype=int)
    pos = np.where(long_on, 1, np.where(long_off, 0, np.nan))
    pos = pd.Series(pos, index=df.index).ffill().fillna(0).astype(int)

    out = df.copy()
    out["DI"] = di
    out["DI_EMA"] = sig
    out["DI_BaseEMA"] = base
    out["Threshold"] = float(threshold)
    out["Position"] = pos
    return out

def backtest_next_open(df, lot_size=1):
    # Execute entries/exits at next candle open
    pos = df["Position"]
    pos_prev = pos.shift(1).fillna(0)

    entries = (pos == 1) & (pos_prev == 0)
    exits = (pos == 0) & (pos_prev == 1)

    entry_price = next_open_prices(df)
    exit_price = next_open_prices(df)

    trades = []
    in_trade = False
    e_time = None
    e_px = None

    for t, is_entry in entries.iteritems():
        if is_entry and not in_trade:
            e_time = t
            e_px = entry_price.loc[t]
            in_trade = True

    # Iterate over times to close at exits
    for t, is_exit in exits.iteritems():
        if is_exit and in_trade:
            x_time = t
            x_px = exit_price.loc[t]
            if pd.notna(e_px) and pd.notna(x_px):
                trades.append((e_time, e_px, x_time, x_px))
            in_trade = False
            e_time = None
            e_px = None

    # If still in trade, ignore open trade (no mark-to-market)
    if len(trades) == 0:
        trade_df = pd.DataFrame(columns=["EntryTime", "Entry", "ExitTime", "Exit"])
    else:
        trade_df = pd.DataFrame(trades, columns=["EntryTime", "Entry", "ExitTime", "Exit"])

    # Shift timestamps to actual execution times: entry/exit at next bar open
    # Our e_time and x_time correspond to signal bar; execution occurs at t+1 bar open, which we already took as price.
    # For clarity, stamp the execution timestamps as the actual next bar timestamps.
    # We'll fetch them as the index position + 1.
    idx = df.index
    idx_pos = {ts: i for i, ts in enumerate(idx)}

    def exec_time(signal_time):
        i = idx_pos.get(signal_time, None)
        if i is None or i + 1 >= len(idx):
            return pd.NaT
        return idx[i + 1]

    if not trade_df.empty:
        trade_df["EntryExecTime"] = trade_df["EntryTime"].apply(exec_time)
        trade_df["ExitExecTime"] = trade_df["ExitTime"].apply(exec_time)

    # Clean invalid (last-bar) trades with NaT execution
    if not trade_df.empty:
        trade_df = trade_df.dropna(subset=["EntryExecTime", "ExitExecTime"])

    if trade_df.empty:
        return trade_df.assign(
            Points=pd.Series(dtype=float),
            PnL=pd.Series(dtype=float)
        )

    trade_df["Points"] = (trade_df["Exit"] - trade_df["Entry"])
    trade_df["PnL"] = trade_df["Points"] * lot_size

    # Audit-friendly columns
    trade_df = trade_df[[
        "EntryExecTime", "Entry", "ExitExecTime", "Exit", "Points", "PnL"
    ]].rename(columns={
        "EntryExecTime": "EntryIST",
        "ExitExecTime": "ExitIST"
    })

    # Ensure IST tz and sort
    trade_df["EntryIST"] = pd.to_datetime(trade_df["EntryIST"]).dt.tz_convert(IST)
    trade_df["ExitIST"] = pd.to_datetime(trade_df["ExitIST"]).dt.tz_convert(IST)
    trade_df = trade_df.sort_values("EntryIST").reset_index(drop=True)
    return trade_df

def attach_settings_to_trades(trades, symbol, timeframe, di_len, sig_len, threshold, lot_size):
    if trades.empty:
        trades["Symbol"] = symbol
        trades["Timeframe"] = timeframe
        trades["DI_Len"] = di_len
        trades["SignalEMA"] = sig_len
        trades["Threshold"] = threshold
        trades["LotSize"] = lot_size
        return trades
    trades.insert(0, "Symbol", symbol)
    trades.insert(1, "Timeframe", timeframe)
    trades.insert(2, "DI_Len", di_len)
    trades.insert(3, "SignalEMA", sig_len)
    trades.insert(4, "Threshold", threshold)
    trades.insert(5, "LotSize", lot_size)
    return trades

def pnl_breakdowns(trades):
    if trades.empty:
        return (
            pd.DataFrame(columns=["Month", "Trades", "PnL"]),
            pd.DataFrame(columns=["Date", "Trades", "PnL"])
        )
    t = trades.copy()
    t["ExitDate"] = trades["ExitIST"].dt.tz_convert(IST).dt.date
    t["ExitMonth"] = trades["ExitIST"].dt.tz_convert(IST).dt.to_period("M").dt.to_timestamp()

    daily = t.groupby("ExitDate").agg(Trades=("PnL", "count"), PnL=("PnL", "sum")).reset_index()
    monthly = t.groupby("ExitMonth").agg(Trades=("PnL", "count"), PnL=("PnL", "sum")).reset_index()
    monthly = monthly.rename(columns={"ExitMonth": "Month"})
    daily = daily.rename(columns={"ExitDate": "Date"})
    return monthly, daily

def max_drawdown(equity_curve):
    if equity_curve.empty:
        return 0.0
    cummax = equity_curve.cummax()
    dd = equity_curve - cummax
    return dd.min()

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="DI Strategy — NIFTY/BANKNIFTY", layout="wide")

st.title("Disparity Index (DI) Strategy — NIFTY & BANKNIFTY")

with st.sidebar:
    st.header("Global settings")
    # Symbols and mapping
    symbols_ui = ["NIFTY 50", "BANKNIFTY"]
    symbol_map = {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
    }
    selected = st.multiselect("Symbols", symbols_ui, default=symbols_ui)

    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=1)
    lookback = st.slider("Lookback days", min_value=5, max_value=730, value=120, step=5)

    st.divider()
    st.subheader("Auto-refresh")
    live_mode = st.toggle("Enable auto-refresh")
    refresh_secs = st.slider("Refresh every (seconds)", 10, 300, 30, step=5)
    if live_mode:
        st.caption("Auto-refresh is active")
        st.experimental_rerun  # hint to Streamlit cloud to prep
        st.autorefresh = st.experimental_rerun  # no-op placeholder
        st.experimental_set_query_params(refresh=int(time.time()))

    st.divider()
    st.caption("Execution uses next-candle open for both entries and exits.")

# Per-symbol parameter defaults
defaults = {
    "NIFTY 50": dict(di_len=29, sig_len=27, threshold=81.0, lot_size=1),
    "BANKNIFTY": dict(di_len=29, sig_len=27, threshold=81.0, lot_size=1),
}

tabs = st.tabs(selected if selected else ["No symbol selected"])

results_summary = []

for i, name in enumerate(selected):
    with tabs[i]:
        yf_symbol = symbol_map[name]
        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])

        with col1:
            di_len = st.number_input(f"{name} DI length", min_value=3, max_value=200, value=defaults[name]["di_len"], step=1, key=f"{name}_di")
        with col2:
            sig_len = st.number_input(f"{name} Signal EMA", min_value=2, max_value=200, value=defaults[name]["sig_len"], step=1, key=f"{name}_sig")
        with col3:
            threshold = st.number_input(f"{name} Threshold", min_value=50.0, max_value=150.0, value=float(defaults[name]["threshold"]), step=0.5, key=f"{name}_thr")
        with col4:
            lot_size = st.number_input(f"{name} Lot size (points→PnL)", min_value=1, max_value=100000, value=defaults[name]["lot_size"], step=1, key=f"{name}_lot")
        with col5:
            st.write("")  # spacer
            csv_file = st.file_uploader(f"{name} CSV (optional)", type=["csv"], key=f"{name}_csv")

        # Data
        if csv_file is not None:
            df = parse_csv(csv_file)
        else:
            df = fetch_yf(yf_symbol, timeframe, lookback)

        if df.empty:
            st.warning(f"No data for {name}. Try CSV upload or adjust timeframe/lookback.")
            continue

        # Strategy compute
        df_sig = generate_signals(df, di_len, sig_len, threshold)
        trades = backtest_next_open(df_sig, lot_size=lot_size)
        trades = attach_settings_to_trades(trades, name, timeframe, di_len, sig_len, threshold, lot_size)

        # Equity curve (realized)
        if trades.empty:
            equity = pd.Series(dtype=float)
        else:
            equity = trades.set_index("ExitIST")["PnL"].cumsum()

        # KPIs
        total_trades = int(trades.shape[0])
        wins = int((trades["PnL"] > 0).sum()) if total_trades else 0
        win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
        net_pnl = float(trades["PnL"].sum()) if total_trades else 0.0
        dd = float(max_drawdown(equity)) if not equity.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Trades", total_trades)
        k2.metric("Win rate %", f"{win_rate:.1f}")
        k3.metric("Net PnL", f"{net_pnl:,.2f}")
        k4.metric("Max Drawdown", f"{dd:,.2f}")

        results_summary.append({
            "Symbol": name,
            "Timeframe": timeframe,
            "Trades": total_trades,
            "Win%": round(win_rate, 1),
            "NetPnL": round(net_pnl, 2),
            "MaxDD": round(dd, 2),
        })

        # Charts
        with st.container():
            c1, c2 = st.columns([2, 1])

            with c1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_sig.index, open=df_sig["Open"], high=df_sig["High"],
                    low=df_sig["Low"], close=df_sig["Close"], name="Price"
                ))
                fig.add_trace(go.Scatter(
                    x=df_sig.index, y=df_sig["DI_BaseEMA"], name=f"EMA({di_len})", line=dict(color="orange", width=1)
                ))

                # Entry/Exit markers
                if not trades.empty:
                    fig.add_trace(go.Scatter(
                        x=trades["EntryIST"], y=trades["Entry"], mode="markers",
                        marker=dict(symbol="triangle-up", color="green", size=9),
                        name="Entry"
                    ))
                    fig.add_trace(go.Scatter(
                        x=trades["ExitIST"], y=trades["Exit"], mode="markers",
                        marker=dict(symbol="triangle-down", color="red", size=9),
                        name="Exit"
                    ))

                fig.update_layout(
                    title=f"{name} — Price & EMA({di_len})",
                    xaxis_title="Time (IST)",
                    yaxis_title="Price",
                    height=520,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_sig.index, y=df_sig["DI"], name="DI", line=dict(color="#1f77b4")))
                fig2.add_trace(go.Scatter(x=df_sig.index, y=df_sig["DI_EMA"], name=f"DI EMA({sig_len})", line=dict(color="#ff7f0e")))
                fig2.add_trace(go.Scatter(
                    x=df_sig.index, y=df_sig["Threshold"], name=f"Threshold {threshold}",
                    line=dict(color="purple", width=1, dash="dash")
                ))
                fig2.update_layout(
                    title="Disparity Index",
                    xaxis_title="Time (IST)",
                    yaxis_title="DI",
                    height=520,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # Trade log
        st.subheader(f"{name} trade log (IST)")
        if trades.empty:
            st.info("No trades for the current settings and lookback.")
        else:
            # Friendly columns order
            view_cols = [
                "Symbol","Timeframe","DI_Len","SignalEMA","Threshold","LotSize",
                "EntryIST","Entry","ExitIST","Exit","Points","PnL"
            ]
            st.dataframe(trades[view_cols], use_container_width=True, height=300)

            # Download
            csv_buf = io.StringIO()
            trades[view_cols].to_csv(csv_buf, index=False)
            st.download_button(
                label=f"Download {name} trades CSV",
                data=csv_buf.getvalue(),
                file_name=f"{name.replace(' ','_')}_DI_trades.csv",
                mime="text/csv"
            )

        # PnL breakdowns
        st.subheader(f"{name} PnL breakdowns")
        monthly, daily = pnl_breakdowns(trades)

        colm, cold = st.columns(2)
        with colm:
            st.markdown("**Monthly PnL**")
            st.dataframe(monthly, use_container_width=True, height=220)
            if not monthly.empty:
                figm = go.Figure(go.Bar(x=monthly["Month"], y=monthly["PnL"], name="Monthly PnL"))
                figm.update_layout(height=260, xaxis_title="Month", yaxis_title="PnL")
                st.plotly_chart(figm, use_container_width=True)
        with cold:
            st.markdown("**Daily PnL**")
            st.dataframe(daily, use_container_width=True, height=220)
            if not daily.empty:
                figd = go.Figure(go.Bar(x=daily["Date"], y=daily["PnL"], name="Daily PnL"))
                figd.update_layout(height=260, xaxis_title="Date", yaxis_title="PnL")
                st.plotly_chart(figd, use_container_width=True)

        st.divider()

        # Equity curve (realized)
        st.subheader(f"{name} equity (realized PnL)")
        if equity.empty:
            st.info("No realized PnL yet.")
        else:
            fige = go.Figure(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="teal")))
            fige.update_layout(height=300, xaxis_title="Time (IST)", yaxis_title="Cumulative PnL")
            st.plotly_chart(fige, use_container_width=True)

# Combined summary
if results_summary:
    st.header("Summary")
    st.dataframe(pd.DataFrame(results_summary), use_container_width=True)
# --- Main ---
for label, symbol in symbols.items():
    st.subheader(f"📈 {label} ({symbol})")
    df = fetch_data(symbol)
    df = calculate_DI(df, length=6, short=9, long=20)

    st.line_chart(df[['Close']])
    st.line_chart(df[['DI_short', 'DI_long']])

    total_return, final_value, max_dd, wins, losses, log_df = run_backtest(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Initial Capital", "₹100000")
    col2.metric("Final Value", f"₹{final_value:,.2f}")
    col3.metric("Total Return", f"{total_return:.2f}%")
    col4.metric("Max Drawdown", f"{max_dd:.2f}%")

    col5, col6 = st.columns(2)
    col5.metric("Winning Trades", wins)
    col6.metric("Losing Trades", losses)

    st.subheader("📜 Trade Log")
    st.dataframe(log_df)

    csv = log_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"Download {label} Trade Log",
        data=csv,
        file_name=f"{label}_trade_log.csv",
        mime="text/csv"
    )
