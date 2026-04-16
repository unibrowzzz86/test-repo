"""Streamlit dashboard for the ES Futures Backtesting System.

Matches the dark-themed trading dashboard layout from the screenshot:
- Top-line metrics (total return, trades, net profit)
- Metric cards (win rate, profit factor, max DD, Sharpe, Sortino, Calmar)
- Avg Win / Avg Loss
- Monthly breakdown bars
- Equity curve and drawdown chart
- Walk-forward analysis results
- Monte Carlo simulation
- Regime analysis
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.engine import Backtester, BacktestConfig
from backtester.costs import CostModel, ES_FUTURES
from backtester.position_sizing import PositionSizer
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import MeanReversion
from backtester.strategies.momentum_breakout import MomentumBreakout
from backtester.strategies.vwap_strategy import VWAPStrategy
from backtester.strategies.ensemble import EnsembleStrategy
from backtester.data.provider import DataProvider
from backtester.analysis.walk_forward import WalkForwardAnalysis
from backtester.analysis.monte_carlo import MonteCarloSimulation
from backtester.analysis.regime import RegimeAnalysis
from backtester.monitoring.monitor import LiveMonitor

# Page config
st.set_page_config(
    page_title="ES Futures Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS to match the screenshot
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        margin: 4px;
        border: 1px solid #2a2a4a;
    }
    .metric-label {
        color: #8888aa;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-value-green {
        color: #00d4aa;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-value-red {
        color: #ff6b6b;
        font-size: 28px;
        font-weight: bold;
    }
    .metric-value-white {
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
    }
    .big-return {
        color: #00d4aa;
        font-size: 48px;
        font-weight: bold;
    }
    .sub-text {
        color: #8888aa;
        font-size: 13px;
    }
    .header-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #2a2a4a;
    }
    .stApp {
        background-color: #0a0a1a;
    }
</style>
""", unsafe_allow_html=True)


def create_metric_card(label, value, color="green"):
    color_class = f"metric-value-{color}"
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{color_class}">{value}</div>
    </div>
    """


def run_dashboard():
    st.title("ES Futures Strategy Backtester")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Capital & Costs")
        initial_capital = st.number_input("Initial Capital ($)", value=50000, step=5000)
        commission = st.number_input("Commission/Contract ($)", value=2.25, step=0.25)
        slippage_ticks = st.number_input("Slippage (ticks)", value=1.0, step=0.5)
        signal_delay = st.number_input("Signal Delay (bars)", value=0, step=1)

        st.subheader("Data")
        data_source = st.selectbox(
            "Data Source",
            ["Generate Intraday ES Data (30min)", "Generate Daily ES Data", "Upload CSV"],
        )
        if data_source in ["Generate Intraday ES Data (30min)", "Generate Daily ES Data"]:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-15"))
            seed = st.number_input("Random Seed", value=42, step=1)
        else:
            uploaded_file = st.file_uploader("Upload OHLCV CSV", type="csv")

        st.subheader("Strategy")
        strategy_choice = st.selectbox(
            "Strategy",
            ["Ensemble (All Combined)", "SMA Crossover", "Mean Reversion",
             "Momentum Breakout", "VWAP Strategy"],
        )

        st.subheader("Risk Management")
        daily_loss_limit = st.number_input("Daily Loss Limit (%)", value=3.0, step=0.5)
        dd_reduce = st.number_input("Reduce Size at DD (%)", value=10.0, step=1.0)
        dd_stop = st.number_input("Stop Trading at DD (%)", value=20.0, step=1.0)

        run_button = st.button("Run Backtest", type="primary", use_container_width=True)

    if run_button:
        with st.spinner("Running backtest..."):
            # Load data
            if data_source == "Generate Intraday ES Data (30min)":
                data = DataProvider.generate_intraday_es_data(
                    start_date=str(start_date),
                    end_date=str(end_date),
                    bar_minutes=30,
                    seed=int(seed),
                )
            elif data_source == "Generate Daily ES Data":
                data = DataProvider.generate_es_data(
                    start_date=str(start_date),
                    end_date=str(end_date),
                    seed=int(seed),
                )
            else:
                if uploaded_file is None:
                    st.error("Please upload a CSV file")
                    return
                data = DataProvider.from_csv(uploaded_file)

            # Configure
            cost_model = CostModel(
                commission_per_contract=commission,
                slippage_ticks=slippage_ticks,
                signal_delay_bars=int(signal_delay),
            )
            sizer = PositionSizer(
                daily_loss_limit_pct=daily_loss_limit,
                drawdown_reduce_pct=dd_reduce,
                drawdown_stop_pct=dd_stop,
            )
            config = BacktestConfig(
                initial_capital=float(initial_capital),
                cost_model=cost_model,
                position_sizer=sizer,
            )

            # Build strategy (tuned for intraday 30-min bars)
            if strategy_choice == "Ensemble (All Combined)":
                strategy = EnsembleStrategy(threshold=0.3)
                strategy.add_strategy(SMACrossover(
                    fast_period=8, slow_period=21, trend_filter_period=50,
                    atr_stop_multiplier=1.5,
                ), 1.0)
                strategy.add_strategy(MeanReversion(
                    rsi_period=10, rsi_oversold=25, rsi_overbought=75,
                    bb_period=15, atr_stop_multiplier=1.5,
                ), 1.0)
                strategy.add_strategy(MomentumBreakout(
                    lookback_period=15, volume_threshold=1.2,
                    atr_stop_multiplier=2.0, max_hold_bars=20,
                ), 0.8)
                strategy.add_strategy(VWAPStrategy(
                    vwap_dev_threshold=1.2, rsi_period=10,
                    atr_stop_multiplier=1.5,
                ), 0.8)
            elif strategy_choice == "SMA Crossover":
                strategy = SMACrossover(
                    fast_period=8, slow_period=21, trend_filter_period=50,
                    atr_stop_multiplier=1.5,
                )
            elif strategy_choice == "Mean Reversion":
                strategy = MeanReversion(
                    rsi_period=10, rsi_oversold=25, rsi_overbought=75,
                    bb_period=15, atr_stop_multiplier=1.5,
                )
            elif strategy_choice == "Momentum Breakout":
                strategy = MomentumBreakout(
                    lookback_period=15, volume_threshold=1.2,
                    atr_stop_multiplier=2.0, max_hold_bars=20,
                )
            else:
                strategy = VWAPStrategy(
                    vwap_dev_threshold=1.2, rsi_period=10,
                    atr_stop_multiplier=1.5,
                )

            # Run backtest
            bt = Backtester(config)
            result = bt.run(strategy, data)
            m = result["metrics"]

            # Store in session state
            st.session_state["result"] = result
            st.session_state["data"] = data
            st.session_state["config"] = config
            st.session_state["strategy"] = strategy

    # Display results
    if "result" not in st.session_state:
        st.info("Configure settings in the sidebar and click 'Run Backtest' to begin.")
        return

    result = st.session_state["result"]
    data = st.session_state["data"]
    config = st.session_state["config"]
    strategy = st.session_state["strategy"]
    m = result["metrics"]

    # ═══════════════════════════════════════════
    # TOP SECTION: Big return number + headline stats
    # ═══════════════════════════════════════════
    st.markdown("---")

    col_main, col_stats = st.columns([2, 1])

    with col_main:
        ret_pct = m.get('total_return_pct', 0)
        ret_sign = "+" if ret_pct >= 0 else ""
        ret_color = "#00d4aa" if ret_pct >= 0 else "#ff6b6b"
        st.markdown(f"""
        <div class="header-card">
            <div class="metric-label">TOTAL RETURN</div>
            <div style="color: {ret_color}; font-size: 48px; font-weight: bold;">{ret_sign}{ret_pct:.1f}%</div>
            <div class="sub-text">${m.get('net_profit', 0):,.0f} on ${m.get('initial_capital', 0):,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        trades_count = m.get("num_trades", 0)
        net_profit = m.get("net_profit", 0)
        st.markdown(f"""
        <div class="header-card">
            <div style="text-align: right;">
                <div class="metric-label">{trades_count} TRADES</div>
                <div class="metric-value-green">${net_profit:,.1f}k</div>
                <div class="sub-text">net profit</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # METRICS ROW 1: Win Rate, Profit Factor, Max DD
    # ═══════════════════════════════════════════
    st.markdown("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(create_metric_card(
            "WIN RATE", f"{m.get('win_rate_pct', 0):.2f}%", "green"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(create_metric_card(
            "PROFIT FACTOR", f"{m.get('profit_factor', 0):.3f}", "green"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(create_metric_card(
            "MAX DD", f"{m.get('max_drawdown_pct', 0):.2f}%", "red"
        ), unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # METRICS ROW 2: Sharpe, Sortino, Calmar
    # ═══════════════════════════════════════════
    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown(create_metric_card(
            "SHARPE", f"{m.get('sharpe_ratio', 0):.3f}", "white"
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(create_metric_card(
            "SORTINO", f"{m.get('sortino_ratio', 0):.2f}", "white"
        ), unsafe_allow_html=True)
    with c6:
        st.markdown(create_metric_card(
            "CALMAR", f"{m.get('calmar_ratio', 0):.2f}", "white"
        ), unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # METRICS ROW 3: Avg Win, Avg Loss
    # ═══════════════════════════════════════════
    c7, c8 = st.columns(2)
    with c7:
        st.markdown(create_metric_card(
            "AVG WIN", f"${m.get('avg_win', 0):,.0f}", "green"
        ), unsafe_allow_html=True)
    with c8:
        st.markdown(create_metric_card(
            "AVG LOSS", f"${m.get('avg_loss', 0):,.0f}", "red"
        ), unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # MONTHLY BREAKDOWN
    # ═══════════════════════════════════════════
    monthly = m.get("monthly_breakdown", {})
    if monthly:
        st.markdown("### Monthly Breakdown")

        months_data = []
        for month_key, vals in sorted(monthly.items()):
            months_data.append({
                "Month": month_key,
                "P&L": vals.get("pnl", 0),
                "Win Rate": vals.get("win_rate", 0),
                "Trades": vals.get("trades", 0),
            })

        if months_data:
            mdf = pd.DataFrame(months_data)
            colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in mdf["P&L"]]

            fig_monthly = go.Figure(data=[
                go.Bar(
                    x=mdf["Month"],
                    y=mdf["P&L"],
                    marker_color=colors,
                    text=[f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}" for v in mdf["P&L"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.0f}<br>Win Rate: %{customdata[0]:.1f}%<br>Trades: %{customdata[1]}<extra></extra>",
                    customdata=list(zip(mdf["Win Rate"], mdf["Trades"])),
                ),
            ])
            fig_monthly.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0a1a",
                plot_bgcolor="#1a1a2e",
                height=300,
                margin=dict(l=40, r=40, t=20, b=40),
                yaxis_title="P&L ($)",
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

    # ═══════════════════════════════════════════
    # EQUITY CURVE & DRAWDOWN
    # ═══════════════════════════════════════════
    st.markdown("### Equity Curve")

    equity = m.get("equity_curve")
    if equity is not None and len(equity) > 0:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="Equity",
            line=dict(color="#00d4aa", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
        ))
        fig_eq.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0a1a",
            plot_bgcolor="#1a1a2e",
            height=400,
            margin=dict(l=40, r=40, t=20, b=40),
            yaxis_title="Equity ($)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown chart
    dd_series = m.get("drawdown_series")
    if dd_series is not None and len(dd_series) > 0:
        st.markdown("### Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values,
            mode="lines", name="Drawdown",
            line=dict(color="#ff6b6b", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.2)",
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0a1a",
            plot_bgcolor="#1a1a2e",
            height=250,
            margin=dict(l=40, r=40, t=20, b=40),
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ═══════════════════════════════════════════
    # TRADE DISTRIBUTION
    # ═══════════════════════════════════════════
    trades_df = result.get("trades")
    if trades_df is not None and not trades_df.empty:
        st.markdown("### Trade Distribution")

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=trades_df["pnl"],
            nbinsx=40,
            marker_color="#00d4aa",
            opacity=0.7,
        ))
        fig_dist.add_vline(x=0, line_dash="dash", line_color="#ff6b6b")
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0a1a",
            plot_bgcolor="#1a1a2e",
            height=300,
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis_title="Trade P&L ($)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ═══════════════════════════════════════════
    # ADVANCED ANALYSIS TABS
    # ═══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## Robustness Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Walk-Forward", "Monte Carlo", "Regime Analysis", "Go-Live Checklist"
    ])

    with tab1:
        st.markdown("### Walk-Forward Analysis")
        st.caption("Tests if the strategy works on unseen data by splitting into in-sample/out-of-sample periods.")

        if st.button("Run Walk-Forward Analysis"):
            with st.spinner("Running walk-forward analysis..."):
                wfa = WalkForwardAnalysis(
                    in_sample_days=40,
                    out_of_sample_days=15,
                    step_days=15,
                )
                wf_result = wfa.run(strategy, data, config)

                if "error" in wf_result:
                    st.warning(wf_result["error"])
                else:
                    wfc1, wfc2 = st.columns(2)
                    with wfc1:
                        st.metric("Avg OOS Return", f"{wf_result['avg_oos_return_pct']}%")
                        st.metric("Avg OOS Sharpe", f"{wf_result['avg_oos_sharpe']}")
                        st.metric("OOS Win Rate", f"{wf_result['avg_oos_win_rate']}%")
                    with wfc2:
                        st.metric("WF Efficiency", f"{wf_result['walk_forward_efficiency_pct']}%")
                        st.metric("Profitable Windows", f"{wf_result['oos_profitable_windows_pct']}%")
                        st.metric("Windows Tested", wf_result["num_windows"])

                    oos_df = wf_result.get("out_of_sample_results")
                    if oos_df is not None:
                        st.dataframe(oos_df, use_container_width=True)

    with tab2:
        st.markdown("### Monte Carlo Simulation")
        st.caption("Shuffles trade order 1000x to separate skill from luck.")

        if st.button("Run Monte Carlo"):
            with st.spinner("Running 1000 simulations..."):
                mc = MonteCarloSimulation(num_simulations=1000)
                mc_result = mc.run(trades_df, config.initial_capital)

                if "error" in mc_result:
                    st.warning(mc_result["error"])
                else:
                    mcc1, mcc2, mcc3 = st.columns(3)
                    with mcc1:
                        st.metric("Probability of Profit", f"{mc_result['probability_of_profit']}%")
                        st.metric("Risk of Ruin", f"{mc_result['risk_of_ruin_pct']}%")
                    with mcc2:
                        st.metric("Median Final Equity", f"${mc_result['final_equity_median']:,.0f}")
                        st.metric("5th Percentile", f"${mc_result['final_equity_percentiles'][5]:,.0f}")
                    with mcc3:
                        st.metric("95th %ile Max DD", f"{mc_result['max_drawdown_95th_pct']}%")
                        st.metric("Median Sharpe", f"{mc_result['sharpe_median']}")

                    # Distribution plot
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(
                        x=mc_result["final_equity_distribution"],
                        nbinsx=50,
                        marker_color="#00d4aa",
                        opacity=0.7,
                    ))
                    fig_mc.add_vline(x=config.initial_capital, line_dash="dash",
                                     line_color="#ff6b6b", annotation_text="Break Even")
                    fig_mc.add_vline(x=mc_result["original_final_equity"], line_dash="dash",
                                     line_color="#00d4aa", annotation_text="Actual")
                    fig_mc.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0a1a",
                        plot_bgcolor="#1a1a2e",
                        height=350,
                        title="Final Equity Distribution (1000 simulations)",
                        xaxis_title="Final Equity ($)",
                        yaxis_title="Count",
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

    with tab3:
        st.markdown("### Regime Analysis")
        st.caption("How does the strategy perform in different market conditions?")

        if st.button("Run Regime Analysis"):
            with st.spinner("Analyzing regimes..."):
                ra = RegimeAnalysis()
                regime_result = ra.analyze(trades_df, data)

                if regime_result:
                    regime_rows = []
                    for regime, stats in regime_result.items():
                        regime_rows.append({
                            "Regime": regime.replace("_", " ").title(),
                            "Trades": stats.get("num_trades", 0),
                            "Total P&L": f"${stats.get('total_pnl', 0):,.0f}",
                            "Win Rate": f"{stats.get('win_rate', 0):.1f}%",
                            "Avg P&L": f"${stats.get('avg_pnl', 0):,.0f}",
                            "Profit Factor": f"{stats.get('profit_factor', 0):.2f}",
                            "% of Trades": f"{stats.get('pct_of_trades', 0):.1f}%",
                        })
                    st.dataframe(pd.DataFrame(regime_rows), use_container_width=True)

    with tab4:
        st.markdown("### Go-Live Readiness Checklist")
        st.caption("From the r/algotrading wisdom: everything you need before going live.")

        monitor = LiveMonitor(
            expected_win_rate=m.get("win_rate_pct", 55),
            expected_profit_factor=m.get("profit_factor", 1.5),
        )
        checklist = monitor.get_checklist()

        for section, items in checklist.items():
            st.markdown(f"**{section.replace('_', ' ').title()}**")
            for item, detail in items.items():
                st.checkbox(
                    f"{item.replace('_', ' ').title()}: {detail}",
                    key=f"check_{section}_{item}",
                )

    # ═══════════════════════════════════════════
    # DETAILED METRICS TABLE
    # ═══════════════════════════════════════════
    st.markdown("---")
    with st.expander("Full Metrics Detail"):
        detail_metrics = {k: v for k, v in m.items()
                        if k not in ["equity_curve", "drawdown_series", "returns",
                                    "monthly_breakdown", "yearly_breakdown"]}
        st.json(detail_metrics)

    with st.expander("Trade Log"):
        if trades_df is not None and not trades_df.empty:
            display_df = trades_df.copy()
            display_df["entry_date"] = display_df["entry_date"].dt.strftime("%Y-%m-%d")
            display_df["exit_date"] = display_df["exit_date"].dt.strftime("%Y-%m-%d")
            display_df["direction"] = display_df["direction"].map({1: "LONG", -1: "SHORT"})
            st.dataframe(display_df, use_container_width=True)

    with st.expander("Configuration"):
        st.json(result.get("config", {}))


if __name__ == "__main__":
    run_dashboard()
