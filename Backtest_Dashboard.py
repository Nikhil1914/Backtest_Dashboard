import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

from fyers_apiv3 import fyersModel
import plotly.express as px

# ========================= FYERS SETUP =========================

# Make sure access.txt contains your access token (APPID:token)
with open("access.txt", "r") as a:
    access_token = a.read().strip()

# Put your client_id here
client_id = "KB4YGO9V7J-100"  # change if needed

fyers = fyersModel.FyersModel(
    client_id=client_id,
    is_async=False,
    token=access_token,
    log_path=""
)

# ==================== SYMBOL BUILD FUNCTION ====================

def build_fyers_symbol(segment, exchange, symbol, year=None, month=None, day=None, strike=None, opt_type=None):
    if segment == "Equity":
        return f"{exchange}:{symbol}-EQ"
    elif segment == "Index":
        return f"{exchange}:{symbol}-INDEX"
    elif segment == "Equity Futures":
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}FUT"
    elif segment in ["Currency Futures", "Commodity Futures"]:
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}FUT"
    elif segment in [
        "Equity Options (Monthly Expiry)",
        "Currency Options (Monthly Expiry)",
        "Commodity Options (Monthly Expiry)",
    ]:
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}{strike}{opt_type}"
    elif segment in ["Equity Options (Weekly Expiry)", "Currency Options (Weekly Expiry)"]:
        return f"{exchange}:{symbol}{str(year)[-2:]}{month.upper()}{day:02d}{strike}{opt_type}"
    else:
        raise ValueError("Unsupported segment type")

# =================== HISTORICAL DATA FETCH =====================

def fetch_data(symbol, start_date, end_date, resolution="60"):
    """
    Fetch historical OHLCV(+OI) from Fyers in chunks.
    Index returned in IST.
    """
    df = pd.DataFrame()

    resolution_minutes = {
        "1": 1, "2": 2, "3": 3, "5": 5, "10": 10, "15": 15,
        "20": 20, "30": 30, "60": 60, "120": 120, "240": 240
    }

    if resolution in resolution_minutes:
        chunk_days = 100
    elif resolution in ["D", "1D"]:
        chunk_days = 366
    elif resolution in ["W", "M"]:
        chunk_days = 365 * 3
    else:
        st.error("Resolution not supported for long-range.")
        return df

    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + dt.timedelta(days=chunk_days - 1), end_date)

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": 1,
            "range_from": current_start.strftime("%Y-%m-%d"),
            "range_to": current_end.strftime("%Y-%m-%d"),
            "cont_flag": 1,
            "oi_flag": 1,
        }

        try:
            response = fyers.history(params)
            if "candles" in response and response["candles"]:
                first_len = len(response["candles"][0])
                if first_len == 7:
                    chunk = pd.DataFrame(
                        response["candles"],
                        columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"],
                    )
                elif first_len == 6:
                    chunk = pd.DataFrame(
                        response["candles"],
                        columns=["Date", "Open", "High", "Low", "Close", "Volume"],
                    )
                    chunk["OI"] = None
                else:
                    st.error(f"Unexpected candle format from {current_start} to {current_end}")
                    current_start = current_end + dt.timedelta(days=1)
                    continue

                chunk["Date"] = pd.to_datetime(chunk["Date"], unit="s")
                chunk["Date"] = (
                    chunk["Date"]
                    .dt.tz_localize("UTC")
                    .dt.tz_convert("Asia/Kolkata")
                    .dt.tz_localize(None)
                )
                chunk = chunk.set_index("Date")
                df = pd.concat([df, chunk])
        except Exception as e:
            st.error(f"Error fetching from {current_start} to {current_end}: {e}")

        current_start = current_end + dt.timedelta(days=1)

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df

# ======================= STRATEGY: EMA/SMA CROSSOVER ===========

def add_moving_averages(df, ma_type, fast_period, slow_period):
    df = df.copy()
    price = df["Close"]

    if ma_type == "SMA":
        df["fast_ma"] = price.rolling(fast_period).mean()
        df["slow_ma"] = price.rolling(slow_period).mean()
    else:  # EMA
        df["fast_ma"] = price.ewm(span=fast_period, adjust=False).mean()
        df["slow_ma"] = price.ewm(span=slow_period, adjust=False).mean()

    # Signal markers (crossovers)
    df["signal_raw"] = 0
    df.loc[
        (df["fast_ma"] > df["slow_ma"])
        & (df["fast_ma"].shift(1) <= df["slow_ma"].shift(1)),
        "signal_raw",
    ] = 1  # bullish crossover
    df.loc[
        (df["fast_ma"] < df["slow_ma"])
        & (df["fast_ma"].shift(1) >= df["slow_ma"].shift(1)),
        "signal_raw",
    ] = -1  # bearish crossover

    return df


def calc_level(entry_price, level_type, level_value, direction):
    """Compute TP/SL price given type & value."""
    if level_type == "Points":
        if direction == "long":
            tp_price = entry_price + level_value
            sl_price = entry_price - level_value
        else:
            tp_price = entry_price - level_value
            sl_price = entry_price + level_value
    else:  # Percent
        tp_factor = 1 + (level_value / 100.0)
        sl_factor = 1 - (level_value / 100.0)
        if direction == "long":
            tp_price = entry_price * tp_factor
            sl_price = entry_price * sl_factor
        else:
            tp_price = entry_price * (2 - tp_factor)
            sl_price = entry_price * (2 - sl_factor)
    return tp_price, sl_price


def backtest_ma_crossover(
    df,
    ma_type="EMA",
    fast_period=9,
    slow_period=21,
    sl_type="Points",
    sl_value=50.0,
    tp_type="Points",
    tp_value=100.0,
):
    """
    Returns:
        trades_df: per-trade results with Entry/Exit time, prices, PnL etc.
    Logic:
        - Long only
        - Entry at bar open after bullish crossover
        - Exit at TP/SL or on crossover in opposite direction
        - No overnight: exit on day change at prev bar close
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy().sort_index()

    # Filter to regular session (intraday)
    df = df.between_time("09:15", "15:30")
    if df.empty:
        return pd.DataFrame()

    df = add_moving_averages(df, ma_type, fast_period, slow_period)

    trades = []
    in_trade = False
    direction = "long"
    entry_price = None
    entry_time = None
    tp_price = None
    sl_price = None
    trade_date = None

    prev_row = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        ts = df.index[i]
        prev_ts = df.index[i - 1]

        signal_prev = prev_row["signal_raw"]

        # Exit on day change (no overnight)
        if in_trade and ts.date() != trade_date:
            exit_price = prev_row["Close"]
            exit_time = prev_ts
            pnl_points = exit_price - entry_price
            trades.append(
                {
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": direction,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL Points": pnl_points,
                    "Return %": (pnl_points / entry_price) * 100.0,
                    "Exit Reason": "EOD",
                }
            )
            in_trade = False
            entry_price = entry_time = tp_price = sl_price = trade_date = None

        # Manage open trade: check SL/TP on current bar
        if in_trade:
            bar_high = row["High"]
            bar_low = row["Low"]

            if direction == "long":
                hit_tp = bar_high >= tp_price
                hit_sl = bar_low <= sl_price

                if hit_sl and hit_tp:
                    exit_price = sl_price  # conservative
                    exit_reason = "SL&TP same bar (SL priority)"
                elif hit_tp:
                    exit_price = tp_price
                    exit_reason = "Target"
                elif hit_sl:
                    exit_price = sl_price
                    exit_reason = "Stop Loss"
                else:
                    exit_price = None
                    exit_reason = None

                if exit_price is not None:
                    exit_time = ts
                    pnl_points = exit_price - entry_price
                    trades.append(
                        {
                            "Entry Time": entry_time,
                            "Exit Time": exit_time,
                            "Direction": direction,
                            "Entry Price": entry_price,
                            "Exit Price": exit_price,
                            "PnL Points": pnl_points,
                            "Return %": (pnl_points / entry_price) * 100.0,
                            "Exit Reason": exit_reason,
                        }
                    )
                    in_trade = False
                    entry_price = entry_time = tp_price = sl_price = trade_date = None
                    continue

        # New entry after bullish crossover
        if (not in_trade) and (signal_prev == 1):
            entry_price = row["Open"]
            entry_time = ts
            trade_date = ts.date()
            direction = "long"
            tp_price, sl_price = calc_level(entry_price, tp_type, tp_value, direction)
            in_trade = True

    # Close trade at last bar if still open
    if in_trade:
        last_row = df.iloc[-1]
        last_ts = df.index[-1]
        exit_price = last_row["Close"]
        exit_time = last_ts
        pnl_points = exit_price - entry_price
        trades.append(
            {
                "Entry Time": entry_time,
                "Exit Time": exit_time,
                "Direction": direction,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "PnL Points": pnl_points,
                "Return %": (pnl_points / entry_price) * 100.0,
                "Exit Reason": "Last bar",
            }
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["Entry Time"] = pd.to_datetime(trades_df["Entry Time"])
        trades_df["Exit Time"] = pd.to_datetime(trades_df["Exit Time"])
        trades_df["Day"] = trades_df["Exit Time"].dt.date

    return trades_df

# =================== BACKTEST RESULTS HELPERS ===================

def compute_equity_from_pnl(bt_df, initial_capital=100000, pnl_col="pnl"):
    df = bt_df.copy()
    ret = df[pnl_col].fillna(0.0)
    equity_curve = initial_capital * (1 + ret).cumprod()
    equity_curve.name = "Equity"
    daily_equity = equity_curve.resample("1D").last().dropna()
    daily_equity.name = "Equity"
    return equity_curve, daily_equity


def daily_pnl_from_equity(daily_equity):
    df = daily_equity.to_frame("Equity").copy()
    df["DayPnL"] = df["Equity"].diff().fillna(0.0)
    return df


def summarize_backtest(trades_df, daily_equity, initial_capital=100000, trading_days_per_year=252):
    summary = {}

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

    if not trades_df.empty:
        n_trades = len(trades_df)
        gross_ret = trades_df["GrossReturn"]
        wins = (gross_ret > 0).sum()
        losses = (gross_ret < 0).sum()
        win_rate = wins / n_trades * 100.0

        avg_win = gross_ret[gross_ret > 0].mean()
        avg_loss = gross_ret[gross_ret < 0].mean()
        profit_factor = (
            gross_ret[gross_ret > 0].sum() / -gross_ret[gross_ret < 0].sum()
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

# ======================= STREAMLIT APP ==========================

def main():
    st.set_page_config(page_title="Fyers EMA Backtest Dashboard", layout="wide")
    st.title("📊 Fyers Strategy Backtest Dashboard")

    st.sidebar.header("Strategy & Market Settings")

    # -------- Strategy selector (more can be added later) --------
    strategy = st.sidebar.selectbox("Select Strategy", ["EMA Crossover"])

    # ---- Shared high-level settings ----
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000, step=5000)

    # ===== EMA CROSSOVER STRATEGY UI =====
    if strategy == "EMA Crossover":
        segment = st.sidebar.selectbox(
            "Segment",
            [
                "Index",
                "Equity",
                "Equity Futures",
                "Equity Options (Monthly Expiry)",
                "Equity Options (Weekly Expiry)",
                "Currency Futures",
                "Currency Options (Monthly Expiry)",
                "Currency Options (Weekly Expiry)",
                "Commodity Futures",
                "Commodity Options (Monthly Expiry)",
            ],
        )

        exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE", "MCX", "CDS", "NFO"])
        symbol = st.sidebar.text_input("Symbol", value="NIFTY")

        year = month = day = strike = opt_type = None
        if segment not in ["Index", "Equity"]:
            current_year = dt.date.today().year
            year = st.sidebar.selectbox(
                "Year",
                list(range(2017, current_year + 2)),
                index=min(5, current_year + 1 - 2017),
            )
            month = st.sidebar.selectbox(
                "Month",
                ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
                index=dt.date.today().month - 1,
            )
            if "Weekly" in segment:
                day = st.sidebar.number_input(
                    "Day", min_value=1, max_value=31, value=min(dt.date.today().day, 28)
                )
            if "Options" in segment:
                strike = st.sidebar.text_input("Strike Price", value="")
                opt_type = st.sidebar.selectbox("Option Type", ["", "CE", "PE"])
        else:
            st.sidebar.markdown("**Note:** No expiry/strike required for Index/Equity.")

        # Date range & resolution
        min_date = dt.date(2017, 7, 3)
        max_date = dt.date.today()
        start_date = st.sidebar.date_input(
            "Start Date",
            value=max_date - dt.timedelta(days=30),
            min_value=min_date,
            max_value=max_date,
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

        resolution = st.sidebar.selectbox(
            "Resolution (Timeframe)",
            ["1", "2", "3", "5", "10", "15", "20", "30", "60", "120", "240", "D"],
            index=9,  # 60m default
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("MA / EMA Settings")

        ma_type = st.sidebar.selectbox("MA Type", ["SMA", "EMA"], index=1)
        period_options = [5, 8, 9, 10, 13, 20, 21, 34, 50, 100, 200]
        fast_period = st.sidebar.selectbox("Fast MA Period", period_options, index=2)  # 9
        slow_period = st.sidebar.selectbox("Slow MA Period", period_options, index=6)  # 50/100/200 etc.

        if slow_period <= fast_period:
            st.sidebar.warning("Slow period should be greater than fast period.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Target & Stop Loss")

        tp_type = st.sidebar.selectbox("Target Type", ["Points", "Percent"], index=0)
        tp_value = st.sidebar.number_input(f"Target ({tp_type})", min_value=0.0, value=100.0, step=1.0)

        sl_type = st.sidebar.selectbox("Stop Loss Type", ["Points", "Percent"], index=0)
        sl_value = st.sidebar.number_input(f"Stop Loss ({sl_type})", min_value=0.0, value=50.0, step=1.0)

        st.sidebar.markdown("---")
        run_bt = st.sidebar.button("🚀 Run EMA Crossover Backtest")

        # ===== MAIN CONTENT AREA =====
        if start_date > end_date:
            st.error("Start date must not be after end date.")
            return

        if run_bt:
            fyers_symbol = build_fyers_symbol(segment, exchange, symbol, year, month, day, strike, opt_type)

            with st.spinner(f"Fetching data for {fyers_symbol} ({resolution}) and running backtest..."):
                data = fetch_data(fyers_symbol, start_date, end_date, resolution)

                if data.empty:
                    st.error("No data received from Fyers. Check symbol/segment/date range.")
                    return

                price_df = data[["Open", "High", "Low", "Close"]].copy()

                trades_df = backtest_ma_crossover(
                    price_df,
                    ma_type=ma_type,
                    fast_period=fast_period,
                    slow_period=slow_period,
                    sl_type=sl_type,
                    sl_value=sl_value,
                    tp_type=tp_type,
                    tp_value=tp_value,
                )

                if trades_df.empty:
                    st.warning("No trades generated with selected parameters.")
                    return

                # Build per-bar pnl series: assign trade return at exit bar only
                pnl_series = pd.Series(0.0, index=price_df.index)
                for _, tr in trades_df.iterrows():
                    exit_time = tr["Exit Time"]
                    if exit_time in pnl_series.index:
                        pnl_series.loc[exit_time] += tr["Return %"] / 100.0

                bt_df = pd.DataFrame({"pnl": pnl_series}, index=price_df.index)

                # For summary functions, we expect GrossReturn in decimal
                trades_stats_df = trades_df.copy()
                trades_stats_df["GrossReturn"] = trades_stats_df["Return %"] / 100.0

                equity_curve, daily_equity = compute_equity_from_pnl(
                    bt_df, initial_capital=initial_capital, pnl_col="pnl"
                )
                daily_df = daily_pnl_from_equity(daily_equity)
                summary = summarize_backtest(trades_stats_df, daily_equity, initial_capital)

            # ===================== DASHBOARD LAYOUT =====================

            total_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
            max_dd_pct = summary["MaxDD_pct"]

            col_p1, col_p2 = st.columns([3, 1])
            with col_p1:
                st.subheader(f"EMA Crossover Backtest – {fyers_symbol}")
                st.caption(f"{start_date} → {end_date}")

                fig_eq = px.line(
                    equity_curve,
                    labels={"value": "Equity", "index": "Time"},
                    title=f"Equity Curve (Initial: {initial_capital:,.0f})",
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            with col_p2:
                st.markdown("### P&L & Drawdown")
                st.metric("P&L", f"{total_pnl:,.2f}")
                st.metric(
                    "Max Drawdown (%)",
                    f"{max_dd_pct:.2f}" if not np.isnan(max_dd_pct) else "NA",
                )

            st.markdown("---")

            # ===================== SUMMARY METRICS =====================

            st.subheader("Backtest Summary")

            trading_days = len(daily_equity)
            win_days = (daily_df["DayPnL"] > 0).sum()
            loss_days = (daily_df["DayPnL"] < 0).sum()

            win_trades = (trades_stats_df["GrossReturn"] > 0).sum()
            loss_trades = (trades_stats_df["GrossReturn"] < 0).sum()
            max_profit = daily_df["DayPnL"].max()
            max_loss = daily_df["DayPnL"].min()
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
                st.metric(
                    "Total Return (%)",
                    f"{summary['total_return_pct']:.2f}"
                    if summary["total_return_pct"] is not None
                    else "NA",
                )
                st.metric(
                    "CAGR (%)",
                    f"{summary['CAGR_pct']:.2f}"
                    if summary["CAGR_pct"] is not None
                    else "NA",
                )
            with c4:
                st.metric("Max Profit (Day)", f"{max_profit:,.2f}")
                st.metric("Max Loss (Day)", f"{max_loss:,.2f}")
                st.metric("Avg P&L / Day", f"{avg_per_day:,.2f}")

            st.markdown("---")

            # ===================== DAILY P&L BAR CHART =====================

            st.subheader("Daily P&L")

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

            # ===================== DAY-WISE HEATMAP =====================

            st.subheader("Daywise Breakdown (Heatmap)")

            heat_df = daily_df.copy()
            heat_df["Date"] = heat_df.index
            heat_df["Month"] = heat_df.index.to_period("M").astype(str)
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

            trades_show = trades_df.copy()
            trades_show["GrossPnL_pct"] = trades_show["Return %"]
            trades_show["ExitDate"] = trades_show["Exit Time"].dt.date

            rows_per_page = 20
            total_pages = int(np.ceil(len(trades_show) / rows_per_page)) or 1
            page = st.number_input(
                "Page", min_value=1, max_value=total_pages, value=1, step=1
            )

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page
            page_df = trades_show.iloc[start_idx:end_idx]

            st.dataframe(
                page_df[
                    [
                        "ExitDate",
                        "Entry Time",
                        "Exit Time",
                        "Direction",
                        "Entry Price",
                        "Exit Price",
                        "PnL Points",
                        "Return %",
                        "Exit Reason",
                    ]
                ],
                use_container_width=True,
            )

            csv_bytes = trades_show.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Trades CSV",
                data=csv_bytes,
                file_name=f"{symbol}_EMA_Crossover_Trades.csv",
                mime="text/csv",
            )

        else:
            st.info("Set parameters in the sidebar and click **Run EMA Crossover Backtest**.")


if __name__ == "__main__":
    main()
