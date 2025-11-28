import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ========================= CORE HELPERS =========================

def compute_equity_from_pnl(bt_df, initial_capital=100000, pnl_col="pnl"):
    """Build equity curve & daily equity from per-bar returns."""
    df = bt_df.copy()
    ret = df[pnl_col].fillna(0.0)
    equity_curve = initial_capital * (1 + ret).cumprod()
    equity_curve.name = "Equity"
    daily_equity = equity_curve.resample("1D").last().dropna()
    daily_equity.name = "Equity"
    return equity_curve, daily_equity


def extract_trades_from_signals(bt_df, price_col="Close", signal_col="signal"):
    """
    Convert 0/1 signal into trades.
    Entry: 0 -> 1, Exit: 1 -> 0, close open trade at last bar if needed.
    """
    df = bt_df.copy()
    sig = df[signal_col].fillna(0).astype(int)

    in_trade = False
    entry_time = None
    entry_price = None
    trades = []

    for t in df.index:
        current_sig = sig.loc[t]
        price = df.loc[t, price_col]

        # entry
        if not in_trade and current_sig == 1:
            in_trade = True
            entry_time = t
            entry_price = price

        # exit
        elif in_trade and current_sig == 0:
            exit_time = t
            exit_price = price
            gross_ret = (exit_price - entry_price) / entry_price
            trades.append(
                {
                    "EntryTime": entry_time,
                    "ExitTime": exit_time,
                    "Direction": "LONG",
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "BarsHeld": (exit_time - entry_time).total_seconds() / 60.0,
                    "GrossReturn": gross_ret,
                }
            )
            in_trade = False
            entry_time = entry_price = None

    # close at last bar if still in trade
    if in_trade:
        t_last = df.index[-1]
        exit_price = df.iloc[-1][price_col]
        gross_ret = (exit_price - entry_price) / entry_price
        trades.append(
            {
                "EntryTime": entry_time,
                "ExitTime": t_last,
                "Direction": "LONG",
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "BarsHeld": (t_last - entry_time).total_seconds() / 60.0,
                "GrossReturn": gross_ret,
            }
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["GrossPnL_pct"] = trades_df["GrossReturn"] * 100.0
        trades_df["Day"] = trades_df["ExitTime"].dt.date

    return trades_df


def summarize_backtest(trades_df, daily_equity, initial_capital=100000, trading_days_per_year=252):
    """Create summary metrics dict."""
    summary = {}

    # ----- equity / returns -----
    if len(daily_equity) > 1:
        total_return = daily_equity.iloc[-1] / daily_equity.iloc[0] - 1.0
        n_days = (daily_equity.index[-1] - daily_equity.index[0]).days
        years = n_days / 365.0 if n_days > 0 else 0
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

        daily_ret = daily_equity.pct_change().dropna()
        vol_daily = daily_ret.std()
        sharpe = (daily_ret.mean() / vol_daily) * np.sqrt(trading_days_per_year) if vol_daily > 0 else np.nan

        roll_max = daily_equity.cummax()
        dd = daily_equity / roll_max - 1.0
        max_dd = dd.min()
    else:
        total_return = cagr = sharpe = max_dd = np.nan

    summary["initial_capital"] = initial_capital
    summary["final_equity"] = float(daily_equity.iloc[-1]) if len(daily_equity) else np.nan
    summary["total_return_pct"] = total_return * 100 if not np.isnan(total_return) else np.nan
    summary["CAGR_pct"] = cagr * 100 if not np.isnan(cagr) else np.nan
    summary["Sharpe"] = sharpe
    summary["MaxDD_pct"] = max_dd * 100 if not np.isnan(max_dd) else np.nan

    # ----- trade stats -----
    if not trades_df.empty:
        n_trades = len(trades_df)
        wins = (trades_df["GrossReturn"] > 0).sum()
        losses = (trades_df["GrossReturn"] < 0).sum()
        win_rate = wins / n_trades * 100.0

        avg_win = trades_df.loc[trades_df["GrossReturn"] > 0, "GrossReturn"].mean()
        avg_loss = trades_df.loc[trades_df["GrossReturn"] < 0, "GrossReturn"].mean()
        profit_factor = (
            trades_df.loc[trades_df["GrossReturn"] > 0, "GrossReturn"].sum()
            / -trades_df.loc[trades_df["GrossReturn"] < 0, "GrossReturn"].sum()
            if losses > 0
            else np.nan
        )

        summary["total_trades"] = int(n_trades)
        summary["win_rate_pct"] = win_rate
        summary["avg_win_pct"] = avg_win * 100 if not np.isnan(avg_win) else np.nan
        summary["avg_loss_pct"] = avg_loss * 100 if not np.isnan(avg_loss) else np.nan
        summary["profit_factor"] = profit_factor
    else:
        summary["total_trades"] = 0
        summary["win_rate_pct"] = np.nan
        summary["avg_win_pct"] = np.nan
        summary["avg_loss_pct"] = np.nan
        summary["profit_factor"] = np.nan

    return summary


def daily_pnl_from_equity(daily_equity):
    """Compute daily P&L from daily equity series."""
    df = daily_equity.to_frame("Equity").copy()
    df["DayPnL"] = df["Equity"].diff().fillna(0.0)
    return df


# ========================= DASHBOARD APP =========================

def main():
    st.set_page_config(page_title="Backtest Dashboard", layout="wide")
    st.title("📊 Backtest Dashboard")

    # --------- TOP CONTROLS (like PDF: strategy + date range) ----------
    col_top1, col_top2 = st.columns([3, 2])

    with col_top1:
        strategy_name = st.text_input("Select Strategy", value="Advanced Delta Neutral")

    with col_top2:
        st.write("Backtest Range")
        range_choice = st.radio(
            "",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "All", "Custom"],
            horizontal=True,
        )

    st.markdown("---")

    # --------- FILE UPLOAD + SETTINGS ----------
    with st.sidebar:
        st.header("Backtest Data")
        uploaded = st.file_uploader(
            "Upload backtest CSV (bar-level)",
            type=["csv"],
            help="Must contain Date/Datetime, Close, pnl (return), signal (0/1).",
        )
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
        pnl_col = st.text_input("PnL column name (return)", value="pnl")
        date_col_hint = st.text_input("Date column (Date or Datetime)", value="Date")
        run_btn = st.button("Run Dashboard")

    if not uploaded or not run_btn:
        st.info("Upload a backtest CSV and click **Run Dashboard**.")
        return

    # --------- LOAD & PREPARE DATA ----------
    df = pd.read_csv(uploaded)

    # datetime handling
    if date_col_hint in df.columns:
        dt_col = date_col_hint
    elif "Datetime" in df.columns:
        dt_col = "Datetime"
    elif "date" in df.columns:
        dt_col = "date"
    else:
        st.error("Cannot find a Date/Datetime column. Adjust the sidebar input.")
        return

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    if "Close" not in df.columns:
        st.error("CSV must have a 'Close' column.")
        return

    if pnl_col not in df.columns:
        st.error(f"CSV must have a '{pnl_col}' column for per-bar returns.")
        return

    if "signal" not in df.columns:
        st.error("CSV must have a 'signal' column (0 = flat, 1 = long).")
        return

    # date range filter (on full data first)
    full_start, full_end = df.index[0], df.index[-1]

    if range_choice == "All":
        start_date, end_date = full_start, full_end
    elif range_choice == "1 Month":
        end_date = full_end
        start_date = end_date - pd.DateOffset(months=1)
    elif range_choice == "3 Months":
        end_date = full_end
        start_date = end_date - pd.DateOffset(months=3)
    elif range_choice == "6 Months":
        end_date = full_end
        start_date = end_date - pd.DateOffset(months=6)
    elif range_choice == "1 Year":
        end_date = full_end
        start_date = end_date - pd.DateOffset(years=1)
    elif range_choice == "2 Years":
        end_date = full_end
        start_date = end_date - pd.DateOffset(years=2)
    else:  # Custom
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Custom Start", full_start.date())
        with c2:
            end_date = st.date_input("Custom End", full_end.date())
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

    mask = (df.index >= start_date) & (df.index <= end_date)
    df_range = df.loc[mask].copy()

    if df_range.empty:
        st.error("No data in selected range.")
        return

    # ---- build equity, trades, summaries ----
    equity_curve, daily_equity = compute_equity_from_pnl(df_range, initial_capital, pnl_col=pnl_col)
    trades_df = extract_trades_from_signals(df_range, price_col="Close", signal_col="signal")
    summary = summarize_backtest(trades_df, daily_equity, initial_capital)
    daily_df = daily_pnl_from_equity(daily_equity)

    # ===================== TOP P&L + EQUITY CHART =====================

    total_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
    max_dd_pct = summary["MaxDD_pct"]

    col_p1, col_p2 = st.columns([3, 1])

    with col_p1:
        st.subheader(f"Backtest for **{strategy_name}**")
        st.caption(f"{start_date.date()} → {end_date.date()}")

        fig_eq = px.line(
            equity_curve,
            labels={"value": "Equity", "index": "Time"},
            title=f"Equity Curve (Initial: {initial_capital:,.0f})",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    with col_p2:
        st.markdown("### P&L & Drawdown")
        st.metric("P&L", f"{total_pnl:,.2f}")
        st.metric("Max Drawdown (%)", f"{max_dd_pct:.2f}" if not np.isnan(max_dd_pct) else "NA")

    st.markdown("---")

    # ===================== SUMMARY GRID (like PDF cards) =====================

    st.subheader("Backtest Summary")

    trading_days = len(daily_equity)
    win_days = (daily_df["DayPnL"] > 0).sum()
    loss_days = (daily_df["DayPnL"] < 0).sum()

    if not trades_df.empty:
        win_trades = (trades_df["GrossReturn"] > 0).sum()
        loss_trades = (trades_df["GrossReturn"] < 0).sum()
        max_profit = daily_df["DayPnL"].max()
        max_loss = daily_df["DayPnL"].min()
    else:
        win_trades = loss_trades = max_profit = max_loss = 0

    avg_per_day = daily_df["DayPnL"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trading Days", trading_days)
        st.metric("Win Days", win_days)
        st.metric("Loss Days", loss_days)
    with c2:
        st.metric("Total Trades", summary["total_trades"])
        st.metric("Win Trades", win_trades)
        st.metric("Loss Trades", loss_trades)
    with c3:
        st.metric("Total Return (%)", f"{summary['total_return_pct']:.2f}" if summary["total_return_pct"] is not None else "NA")
        st.metric("CAGR (%)", f"{summary['CAGR_pct']:.2f}" if summary["CAGR_pct"] is not None else "NA")
    with c4:
        st.metric("Max Profit (Day)", f"{max_profit:,.2f}")
        st.metric("Max Loss (Day)", f"{max_loss:,.2f}")
        st.metric("Avg P&L / Day", f"{avg_per_day:,.2f}")

    st.markdown("---")

    # ===================== MAX PROFIT & LOSS BAR CHART =====================

    st.subheader("Max Profit and Loss (Daily P&L)")

    daily_df_plot = daily_df.copy()
    daily_df_plot["Date"] = daily_df_plot.index.date

    fig_bar = px.bar(
        daily_df_plot,
        x="Date",
        y="DayPnL",
        color=(daily_df_plot["DayPnL"] >= 0),
        color_discrete_map={True: "green", False: "red"},
        labels={"DayPnL": "P&L", "Date": "Date"},
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ===================== DAYWISE BREAKDOWN (heatmap-style) =====================

    st.subheader("Daywise Breakdown (Heatmap)")

    heat_df = daily_df.copy()
    heat_df["Date"] = heat_df.index.date
    heat_df["Month"] = heat_df.index.to_period("M").astype(str)

    # simple heatmap: x=day of month, y=month, color=P&L
    heat_df["DayOfMonth"] = heat_df.index.day

    fig_heat = px.density_heatmap(
        heat_df,
        x="DayOfMonth",
        y="Month",
        z="DayPnL",
        color_continuous_scale="RdYlGn",
        labels={"DayPnL": "P&L"},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ===================== TRANSACTION DETAILS =====================

    st.subheader("Transaction Details")

    if trades_df.empty:
        st.info("No trades generated in this period.")
    else:
        trades_df_disp = trades_df.sort_values("ExitTime").copy()
        trades_df_disp["ExitDate"] = trades_df_disp["ExitTime"].dt.date
        trades_df_disp["GrossPnL"] = trades_df_disp["GrossReturn"] * 100  # %; adapt if needed

        # pagination
        rows_per_page = 20
        total_pages = int(np.ceil(len(trades_df_disp) / rows_per_page))
        page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)

        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        page_df = trades_df_disp.iloc[start_idx:end_idx]

        st.dataframe(
            page_df[
                ["ExitDate", "EntryTime", "ExitTime", "Direction",
                 "EntryPrice", "ExitPrice", "BarsHeld", "GrossPnL"]
            ],
            use_container_width=True,
        )

        # export button
        csv_bytes = trades_df_disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Transactions to CSV",
            data=csv_bytes,
            file_name=f"{strategy_name.replace(' ', '_')}_trades.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
