#!/usr/bin/env python3
"""Example: Run a full ES futures backtest with all analysis.

This script demonstrates how to:
1. Generate synthetic ES data (or load your own CSV)
2. Run individual strategies and an ensemble
3. View all performance metrics
4. Run walk-forward analysis
5. Run Monte Carlo simulation
6. Analyze by market regime
7. Generate a go-live checklist

Usage:
    python examples/run_backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
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


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_metrics(m: dict):
    ret = m.get('total_return_pct', 0)
    sign = "+" if ret >= 0 else ""
    print(f"  Total Return:     {sign}{ret:.1f}%")
    print(f"  Net Profit:       ${m.get('net_profit', 0):,.2f}")
    print(f"  Trades:           {m.get('num_trades', 0)}")
    print(f"  Win Rate:         {m.get('win_rate_pct', 0):.2f}%")
    print(f"  Profit Factor:    {m.get('profit_factor', 0):.3f}")
    print(f"  Sharpe Ratio:     {m.get('sharpe_ratio', 0):.3f}")
    print(f"  Sortino Ratio:    {m.get('sortino_ratio', 0):.3f}")
    print(f"  Calmar Ratio:     {m.get('calmar_ratio', 0):.2f}")
    print(f"  Max Drawdown:     {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Avg Win:          ${m.get('avg_win', 0):,.2f}")
    print(f"  Avg Loss:         ${m.get('avg_loss', 0):,.2f}")
    print(f"  Payoff Ratio:     {m.get('payoff_ratio', 0):.3f}")
    print(f"  Max Win Streak:   {m.get('max_win_streak', 0)}")
    print(f"  Max Loss Streak:  {m.get('max_loss_streak', 0)}")
    print(f"  Worst 5-Trade:    ${m.get('worst_5_trade_pnl', 0):,.2f}")


def main():
    print_header("ES FUTURES BACKTESTING SYSTEM")
    print("  Generating synthetic ES futures 30-min intraday data...")

    # Generate intraday data (30-min bars) - this gives enough bars for
    # strategies to generate hundreds of trades, like the OP's 347 trades
    data = DataProvider.generate_intraday_es_data(
        start_date="2025-01-01",
        end_date="2025-04-15",
        initial_price=5950.0,
        bar_minutes=30,
        seed=42,
    )
    print(f"  Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=50000.0,
        cost_model=ES_FUTURES,
        position_sizer=PositionSizer(
            max_position_size=1,
            daily_loss_limit_pct=3.0,
            drawdown_reduce_pct=10.0,
            drawdown_stop_pct=20.0,
        ),
    )

    bt = Backtester(config)

    # ═══════════════════════════════════════════
    # 1. Run individual strategies (tuned for 30-min bars)
    # ═══════════════════════════════════════════
    strategies = [
        SMACrossover(
            fast_period=8, slow_period=21, trend_filter_period=50,
            atr_period=14, atr_stop_multiplier=1.5, use_ema=True,
        ),
        MeanReversion(
            rsi_period=10, rsi_oversold=25, rsi_overbought=75,
            bb_period=15, bb_std=2.0, atr_stop_multiplier=1.5,
        ),
        MomentumBreakout(
            lookback_period=15, volume_threshold=1.2,
            atr_stop_multiplier=2.0, max_hold_bars=20,
        ),
        VWAPStrategy(
            vwap_dev_threshold=1.2, rsi_period=10,
            atr_stop_multiplier=1.5,
        ),
    ]

    all_results = []
    for strat in strategies:
        result = bt.run(strat, data)
        all_results.append(result)
        print_header(f"Strategy: {strat.name}")
        print_metrics(result["metrics"])

    # ═══════════════════════════════════════════
    # 2. Run ensemble strategy
    # ═══════════════════════════════════════════
    ensemble = EnsembleStrategy(threshold=0.3)
    ensemble.add_strategy(SMACrossover(
        fast_period=8, slow_period=21, trend_filter_period=50,
        atr_stop_multiplier=1.5,
    ), weight=1.0)
    ensemble.add_strategy(MeanReversion(
        rsi_period=10, rsi_oversold=25, rsi_overbought=75,
        bb_period=15, atr_stop_multiplier=1.5,
    ), weight=1.0)
    ensemble.add_strategy(MomentumBreakout(
        lookback_period=15, volume_threshold=1.2,
        atr_stop_multiplier=2.0, max_hold_bars=20,
    ), weight=0.8)
    ensemble.add_strategy(VWAPStrategy(
        vwap_dev_threshold=1.2, rsi_period=10,
        atr_stop_multiplier=1.5,
    ), weight=0.8)

    ensemble_result = bt.run(ensemble, data)
    all_results.append(ensemble_result)

    print_header("Strategy: ENSEMBLE (All Combined)")
    print_metrics(ensemble_result["metrics"])

    # ═══════════════════════════════════════════
    # 3. Strategy comparison
    # ═══════════════════════════════════════════
    print_header("STRATEGY COMPARISON")
    comparison = bt.compare(all_results)
    print(comparison.to_string(index=False))

    # ═══════════════════════════════════════════
    # 4. Walk-forward analysis
    # ═══════════════════════════════════════════
    print_header("WALK-FORWARD ANALYSIS (Ensemble)")
    wfa = WalkForwardAnalysis(in_sample_days=200, out_of_sample_days=80, step_days=80)
    wf_result = wfa.run(ensemble, data, config)

    if "error" not in wf_result:
        print(f"  Windows Tested:            {wf_result['num_windows']}")
        print(f"  Avg OOS Return:            {wf_result['avg_oos_return_pct']}%")
        print(f"  Avg OOS Sharpe:            {wf_result['avg_oos_sharpe']}")
        print(f"  WF Efficiency:             {wf_result['walk_forward_efficiency_pct']}%")
        print(f"  Profitable Windows:        {wf_result['oos_profitable_windows_pct']}%")
    else:
        print(f"  {wf_result['error']}")

    # ═══════════════════════════════════════════
    # 5. Monte Carlo simulation
    # ═══════════════════════════════════════════
    print_header("MONTE CARLO SIMULATION (Ensemble)")
    mc = MonteCarloSimulation(num_simulations=1000)
    mc_result = mc.run(ensemble_result["trades"], config.initial_capital)

    if "error" not in mc_result:
        print(f"  Probability of Profit:     {mc_result['probability_of_profit']}%")
        print(f"  Risk of Ruin:              {mc_result['risk_of_ruin_pct']}%")
        print(f"  Median Final Equity:       ${mc_result['final_equity_median']:,.2f}")
        print(f"  5th Percentile Equity:     ${mc_result['final_equity_percentiles'][5]:,.2f}")
        print(f"  95th Percentile Max DD:    {mc_result['max_drawdown_95th_pct']}%")
    else:
        print(f"  {mc_result['error']}")

    # ═══════════════════════════════════════════
    # 6. Regime analysis
    # ═══════════════════════════════════════════
    print_header("REGIME ANALYSIS (Ensemble)")
    ra = RegimeAnalysis()
    regime_result = ra.analyze(ensemble_result["trades"], data)

    for regime, stats in regime_result.items():
        if stats["num_trades"] > 0:
            print(f"  {regime.replace('_', ' ').title():20s}  "
                  f"Trades: {stats['num_trades']:3d}  "
                  f"P&L: ${stats['total_pnl']:>10,.2f}  "
                  f"WR: {stats['win_rate']:5.1f}%  "
                  f"PF: {stats['profit_factor']:.2f}")

    # ═══════════════════════════════════════════
    # 7. Monthly breakdown
    # ═══════════════════════════════════════════
    monthly = ensemble_result["metrics"].get("monthly_breakdown", {})
    if monthly:
        print_header("MONTHLY BREAKDOWN (Ensemble)")
        for month, vals in sorted(monthly.items()):
            pnl = vals.get("pnl", 0)
            wr = vals.get("win_rate", 0)
            trades = vals.get("trades", 0)
            bar = "█" * max(1, int(abs(pnl) / 500))
            sign = "+" if pnl >= 0 else ""
            print(f"  {month}  {bar:30s}  {sign}${pnl:>10,.2f}  {wr:5.1f}%  ({trades} trades)")

    # ═══════════════════════════════════════════
    # 8. Go-live checklist
    # ═══════════════════════════════════════════
    print_header("GO-LIVE READINESS CHECKLIST")
    monitor = LiveMonitor(
        expected_win_rate=ensemble_result["metrics"].get("win_rate_pct", 55),
        expected_profit_factor=ensemble_result["metrics"].get("profit_factor", 1.5),
    )
    checklist = monitor.get_checklist()
    for section, items in checklist.items():
        print(f"\n  [{section.replace('_', ' ').upper()}]")
        for item, detail in items.items():
            print(f"    [ ] {item.replace('_', ' ').title()}: {detail}")

    print_header("DONE")
    print("  Run the dashboard for interactive exploration:")
    print("    streamlit run dashboard/app.py")
    print()


if __name__ == "__main__":
    main()
