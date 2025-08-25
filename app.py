import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pytz

# --- Page Config ---
st.set_page_config(page_title="DI Backtest - Nifty & BankNifty", layout="wide")
st.title("📊 Disparity Index Backtest (Length=6, Short=9, Long=20)")
st.caption("Data: Last 3 years | Capital: ₹100,000 | Timestamps in IST")

# --- Symbols ---
symbols = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}

IST = pytz.timezone('Asia/Kolkata')

# --- Data Fetch ---
@st.cache_data
def fetch_data(symbol):
    end = datetime.now()
    start = end - timedelta(days=3 * 365)
    df = yf.download(symbol, start=start, end=end)

    # Flatten if MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.dropna(inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert(IST)
    return df

# --- DI Calculation ---
def calculate_DI(df, length, short, long):
    close_series = df['Close'].astype(float)
    ema_series = close_series.ewm(span=length, adjust=False).mean()
    df['EMA'] = ema_series
    df['DI'] = ((close_series - ema_series) / ema_series) * 100
    df['DI_short'] = df['DI'].ewm(span=short, adjust=False).mean()
    df['DI_long'] = df['DI'].ewm(span=long, adjust=False).mean()
    df.dropna(inplace=True)
    return df

# --- Backtest Logic ---
def run_backtest(df):
    df['Signal'] = np.where(df['DI_short'] > df['DI_long'], 1.0, 0.0)
    df['Position'] = df['Signal'].diff()

    capital = 100000
    position = 0
    entry_price = None
    trade_log = []
    portfolio = capital
    history = []

    for i in range(len(df)):
        current_date = df.index[i]
        price = df['Close'].iloc[i]

        # Entry
        if df['Position'].iloc[i] == 1.0 and position == 0:
            entry_price = price
            entry_date = current_date
            shares = capital / price
            position = shares

        # Exit
        elif df['Position'].iloc[i] == -1.0 and position > 0:
            exit_price = price
            exit_date = current_date
            pnl = (exit_price - entry_price) * position
            portfolio += pnl
            trade_log.append({
                "Buy Date": entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                "Buy Price": entry_price,
                "Sell Date": exit_date.strftime('%Y-%m-%d %H:%M:%S'),
                "Sell Price": exit_price,
                "PnL": pnl
            })
            position = 0

        current_value = price * position if position > 0 else portfolio
        history.append(current_value)

    final_value = price * position if position > 0 else portfolio
    total_return = (final_value - capital) / capital * 100
    drawdown = pd.Series(history).cummax() - pd.Series(history)
    max_dd = (drawdown.max() / pd.Series(history).cummax().max()) * 100

    wins = len([t for t in trade_log if t['PnL'] > 0])
    losses = len([t for t in trade_log if t['PnL'] < 0])

    return total_return, final_value, max_dd, wins, losses, pd.DataFrame(trade_log)

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
    cross_up = (df[a].shift(1) <= df[b].shift(1)) & (df[a] > df[b])
    cross_dn = (df[a].shift(1) >= df[b].shift(1)) & (df[a] < df[b])

    df.loc[cross_up, "Signal"] = "BUY"
    df.loc[cross_dn, "Signal"] = "SELL"

    # Label last signal state forward-fill for context
    df["SignalState"] = df["Signal"].replace("HOLD", np.nan).ffill().fillna("HOLD")
    return df

# ---------------------------
# Plotting
# ---------------------------
def build_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Price (left axis)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines", line=dict(color="#1f77b4", width=1.6)),
        secondary_y=False,
    )
    # DI lines (right axis)
    for n, color in [(6, "#9467bd"), (9, "#2ca02c"), (20, "#ff7f0e")]:
        if f"DI_{n}" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[f"DI_{n}"], name=f"DI {n}", mode="lines", line=dict(width=1.2, color=color)
                ),
                secondary_y=True,
            )

    # Buy/Sell markers on price
    buys = df[df["Signal"] == "BUY"]
    sells = df[df["Signal"] == "SELL"]

    fig.add_trace(
        go.Scatter(
            x=buys.index, y=buys["Close"],
            mode="markers",
            name="BUY",
            marker=dict(symbol="triangle-up", size=10, color="#16a34a", line=dict(width=0)),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index, y=sells["Close"],
            mode="markers",
            name="SELL",
            marker=dict(symbol="triangle-down", size=10, color="#dc2626", line=dict(width=0)),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=title,
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="DI %", secondary_y=True)
    return fig

# ---------------------------
# Summary helpers
# ---------------------------
def latest_snapshot(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "last_time": "-",
            "last_close": "-",
            "di6": "-",
            "di9": "-",
            "di20": "-",
            "spread": "-",
            "signal": "NO DATA",
        }
    last = df.iloc[-1]
    spread = safe_number(last.get("DI_9", np.nan) - last.get("DI_20", np.nan), 2)
    return {
        "last_time": df.index[-1].strftime("%Y-%m-%d %H:%M IST"),
        "last_close": safe_number(last.get("Close", np.nan), 2),
        "di6": safe_number(last.get("DI_6", np.nan), 2),
        "di9": safe_number(last.get("DI_9", np.nan), 2),
        "di20": safe_number(last.get("DI_20", np.nan), 2),
        "spread": spread,
        "signal": last.get("SignalState", "HOLD"),
    }

def collect_trade_events(df: pd.DataFrame, symbol: str, tf_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["TimeIST", "Symbol", "Timeframe", "Signal", "Close", "DI_9", "DI_20"])
    events = df[df["Signal"].isin(["BUY", "SELL"])].copy()
    if events.empty:
        return pd.DataFrame(columns=["TimeIST", "Symbol", "Timeframe", "Signal", "Close", "DI_9", "DI_20"])
    out = pd.DataFrame({
        "TimeIST": events.index.tz_convert(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Timeframe": tf_label,
        "Signal": events["Signal"],
        "Close": events["Close"].round(2),
        "DI_9": events["DI_9"].round(2),
        "DI_20": events["DI_20"].round(2),
    })
    return out

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings")
show_fast_di = st.sidebar.checkbox("Show DI 6 (fast) line", value=True)
st.sidebar.caption("Signals use DI 9 vs DI 20 cross. DI 6 is visual only.")
st.sidebar.divider()
st.sidebar.caption("Intraday history is limited by Yahoo (usually ~60 days).")

# ---------------------------
# Main UI
# ---------------------------
st.title("📈 Multi‑Timeframe Disparity Index — NIFTY & BANKNIFTY")

master_logs = []  # for combined CSV export across all symbols/timeframes

sym_tabs = st.tabs([s["name"] for s in SYMBOLS])
for s_idx, sym in enumerate(SYMBOLS):
    with sym_tabs[s_idx]:
        st.subheader(f"{sym['name']} ({sym['code']})")
        tf_tabs = st.tabs([lbl for (lbl, _, _) in TIMEFRAMES])

        # Summary row placeholders per timeframe
        summaries = []

        for i, (label, interval, period) in enumerate(TIMEFRAMES):
            with tf_tabs[i]:
                with st.spinner(f"Fetching {label}..."):
                    df = fetch_yahoo(sym["code"], interval=interval, period=period)
                if df.empty:
                    st.warning(f"No data for {sym['name']} — {label}.")
                    continue

                df = add_di(df, lengths=(6, 9, 20))
                df = add_signals(df, fast=9, slow=20)

                # Optionally hide DI_6 if unchecked
                if not show_fast_di and "DI_6" in df.columns:
                    df = df.drop(columns=["DI_6"])

                # Chart
                fig = build_chart(df, title=f"{sym['name']} — {label}")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Snapshot
                snap = latest_snapshot(df)
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Last time (IST)", snap["last_time"])
                col2.metric("Close", snap["last_close"])
                col3.metric("DI 6", snap["di6"] if show_fast_di else "—")
                col4.metric("DI 9", snap["di9"])
                col5.metric("DI 20", snap["di20"])
                col6.metric("DI spread (9-20)", snap["spread"])

                # Signal badge
                sig = snap["signal"]
                if sig == "BUY":
                    st.success(f"Latest signal: BUY  •  {snap['last_time']}")
                elif sig == "SELL":
                    st.error(f"Latest signal: SELL  •  {snap['last_time']}")
                elif sig == "HOLD":
                    st.info(f"Latest signal: HOLD  •  {snap['last_time']}")
                else:
                    st.warning(f"Latest signal: {sig}")

                # Collect trade events for export
                logs = collect_trade_events(df, sym["name"], label)
                if not logs.empty:
                    master_logs.append(logs)

        # Per‑symbol export
        if master_logs:
            symbol_logs = pd.concat([x for x in master_logs if not x.empty], ignore_index=True) if master_logs else pd.DataFrame()
            if not symbol_logs.empty:
                st.download_button(
                    label=f"⬇️ Download signals CSV — {sym['name']}",
                    data=symbol_logs.to_csv(index=False).encode("utf-8"),
                    file_name=f"{sym['name'].replace(' ', '_')}_DI_signals_IST.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# Footer: run info
now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
st.caption(f"Last refreshed: {now_ist}  •  Data source: Yahoo Finance  •  Timestamps: IST")

    return total_return, final_value, max_dd, wins, losses, pd.DataFrame(trade_log)

# --- Main Loop for Symbols ---
for label, symbol in symbols.items():
    st.subheader(f"📈 {label} ({symbol})")

    df = fetch_data(symbol)
    df = calculate_DI(df, length=6, short=9, long=20)

    # Price chart
    st.line_chart(df[['Close']])

    # DI chart
    st.line_chart(df[['DI_short', 'DI_long']])

    # Backtest results
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
