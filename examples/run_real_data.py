#!/usr/bin/env python3
"""Run backtest on REAL ES futures data.

This is the script you actually want to use. It pulls real data
from Yahoo Finance or Polygon.io and runs the full analysis.

Usage:
    # Using Yahoo Finance (free, no API key needed)
    python examples/run_real_data.py --source yahoo

    # Using Polygon.io (better intraday data, needs API key)
    python examples/run_real_data.py --source polygon --api-key YOUR_KEY

    # Using a CSV export from TradingView
    python examples/run_real_data.py --source csv --file path/to/data.csv
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.engine import Backtester, BacktestConfig
from backtester.costs import ES_FUTURES
from backtester.position_sizing import PositionSizer
from backtester.strategies.es_scalper import ESScalper
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import MeanReversion
from backtester.strategies.momentum_breakout import MomentumBreakout
from backtester.strategies.vwap_strategy import VWAPStrategy
from backtester.strategies.ensemble import EnsembleStrategy
from backtester.data.provider import DataProvider
from backtester.analysis.walk_forward import WalkForwardAnalysis
from backtester.analysis.monte_carlo import MonteCarloSimulation
from backtester.analysis.regime import RegimeAnalysis


def load_data(args):
    """Load data from the specified source."""
    if args.source == "yahoo":
        from backtester.data.yahoo import YahooDataProvider
        print(f"Pulling real ES data from Yahoo Finance...")

        if args.intraday:
            data = YahooDataProvider.get_es_intraday(
                interval=args.interval,
                period=args.period,
            )
        else:
            data = YahooDataProvider.get_es_daily(
                start_date=args.start,
                end_date=args.end,
            )

    elif args.source == "polygon":
        from backtester.data.polygon_provider import PolygonDataProvider
        if not args.api_key:
            print("ERROR: Polygon.io requires an API key. Get one free at https://polygon.io")
            sys.exit(1)

        provider = PolygonDataProvider(api_key=args.api_key)
        print(f"Pulling real ES data from Polygon.io...")

        multiplier = int(args.interval.replace("m", "").replace("min", ""))
        data = provider.get_es_bars(
            start_date=args.start,
            end_date=args.end,
            timespan="minute",
            multiplier=multiplier,
        )

    elif args.source == "csv":
        if not args.file:
            print("ERROR: --file required for CSV source")
            sys.exit(1)
        print(f"Loading data from {args.file}...")
        data = DataProvider.from_csv(args.file)

    else:
        print(f"Unknown source: {args.source}")
        sys.exit(1)

    return data


def build_strategy(name: str):
    """Build a strategy by name."""
    if name == "scalper":
        return ESScalper(
            fast_ema=5, slow_ema=13, rsi_period=6,
            atr_stop_mult=1.0, atr_target_mult=2.5,
            vwap_dev_entry=0.8, max_hold_bars=60, cooldown_bars=2,
        )
    elif name == "ensemble":
        e = EnsembleStrategy(threshold=0.3)
        e.add_strategy(SMACrossover(fast_period=8, slow_period=21, atr_stop_multiplier=1.5), 1.0)
        e.add_strategy(MeanReversion(rsi_period=10, bb_period=15, atr_stop_multiplier=1.5), 1.0)
        e.add_strategy(MomentumBreakout(lookback_period=15, atr_stop_multiplier=2.0), 0.8)
        e.add_strategy(VWAPStrategy(vwap_dev_threshold=1.2, atr_stop_multiplier=1.5), 0.8)
        return e
    elif name == "sma":
        return SMACrossover(fast_period=8, slow_period=21, atr_stop_multiplier=1.5)
    elif name == "mean_reversion":
        return MeanReversion(rsi_period=10, bb_period=15, atr_stop_multiplier=1.5)
    elif name == "momentum":
        return MomentumBreakout(lookback_period=15, atr_stop_multiplier=2.0)
    elif name == "vwap":
        return VWAPStrategy(vwap_dev_threshold=1.2, atr_stop_multiplier=1.5)
    else:
        print(f"Unknown strategy: {name}")
        sys.exit(1)


def print_results(m: dict, name: str):
    ret = m.get('total_return_pct', 0)
    sign = '+' if ret >= 0 else ''

    print(f"\n{'='*60}")
    print(f"  RESULTS: {name}")
    print(f"{'='*60}")
    print(f"  Total Return:     {sign}{ret:.1f}%")
    print(f"  Net Profit:       ${m.get('net_profit', 0):,.2f}")
    print(f"  Final Equity:     ${m.get('final_equity', 0):,.2f}")
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

    monthly = m.get('monthly_breakdown', {})
    if monthly:
        print(f"\n  MONTHLY:")
        for month, vals in sorted(monthly.items()):
            pnl = vals.get('pnl', 0)
            wr = vals.get('win_rate', 0)
            t = vals.get('trades', 0)
            s = '+' if pnl >= 0 else ''
            print(f"    {month}  {s}${pnl:>10,.2f}  WR:{wr:5.1f}%  ({t} trades)")


def main():
    parser = argparse.ArgumentParser(description="Run ES backtest on real data")
    parser.add_argument("--source", choices=["yahoo", "polygon", "csv"], default="yahoo")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-04-15")
    parser.add_argument("--interval", default="30m", help="Bar interval: 1m, 5m, 15m, 30m, 1h")
    parser.add_argument("--period", default="60d", help="Lookback period for intraday")
    parser.add_argument("--intraday", action="store_true", default=True)
    parser.add_argument("--strategy", default="scalper",
                       choices=["scalper", "ensemble", "sma", "mean_reversion", "momentum", "vwap"])
    parser.add_argument("--capital", type=float, default=50000)
    parser.add_argument("--run-analysis", action="store_true", default=True,
                       help="Run walk-forward and Monte Carlo analysis")

    args = parser.parse_args()

    # Load data
    data = load_data(args)
    print(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # Configure
    config = BacktestConfig(
        initial_capital=args.capital,
        cost_model=ES_FUTURES,
        position_sizer=PositionSizer(
            max_position_size=1,
            daily_loss_limit_pct=3.0,
            drawdown_reduce_pct=10.0,
            drawdown_stop_pct=20.0,
        ),
    )

    # Run backtest
    strategy = build_strategy(args.strategy)
    bt = Backtester(config)
    result = bt.run(strategy, data)
    print_results(result["metrics"], strategy.name)

    # Run robustness analysis
    if args.run_analysis and result["metrics"].get("num_trades", 0) >= 10:
        print(f"\n{'='*60}")
        print("  WALK-FORWARD ANALYSIS")
        print(f"{'='*60}")
        wfa = WalkForwardAnalysis(in_sample_days=200, out_of_sample_days=80, step_days=80)
        wf = wfa.run(strategy, data, config)
        if "error" not in wf:
            print(f"  Windows: {wf['num_windows']} | OOS Return: {wf['avg_oos_return_pct']}%")
            print(f"  WF Efficiency: {wf['walk_forward_efficiency_pct']}%")
            print(f"  Profitable windows: {wf['oos_profitable_windows_pct']}%")

        print(f"\n{'='*60}")
        print("  MONTE CARLO (1000 simulations)")
        print(f"{'='*60}")
        mc = MonteCarloSimulation(num_simulations=1000)
        mc_r = mc.run(result["trades"], args.capital)
        if "error" not in mc_r:
            print(f"  P(Profit): {mc_r['probability_of_profit']}%")
            print(f"  Risk of Ruin: {mc_r['risk_of_ruin_pct']}%")
            print(f"  Median Equity: ${mc_r['final_equity_median']:,.0f}")
            print(f"  95th%ile Max DD: {mc_r['max_drawdown_95th_pct']}%")

        print(f"\n{'='*60}")
        print("  REGIME ANALYSIS")
        print(f"{'='*60}")
        ra = RegimeAnalysis()
        reg = ra.analyze(result["trades"], data)
        for regime, stats in reg.items():
            if stats["num_trades"] > 0:
                print(f"  {regime:20s}  T:{stats['num_trades']:3d}  "
                      f"P&L:${stats['total_pnl']:>9,.0f}  "
                      f"WR:{stats['win_rate']:5.1f}%  PF:{stats['profit_factor']:.2f}")

    print(f"\n{'='*60}")
    print("  Launch dashboard: streamlit run dashboard/app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
