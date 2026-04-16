#!/usr/bin/env python3
"""Run the tick-streaming live trading engine.

This is the production entry point for tick-by-tick trading.

Modes:
    --ib-paper    Connect to IB paper trading, stream live ticks
    --ib-live     Connect to IB live account (REAL MONEY)
    --sim         Simulate with generated ticks (no IB needed)

Examples:
    # Paper trade with IB, 5-minute bars from tick stream
    python examples/run_streaming.py --ib-paper --bar-seconds 300

    # Paper trade with IB, 1-minute bars (more trades)
    python examples/run_streaming.py --ib-paper --bar-seconds 60

    # Simulate locally (no IB needed) to test the pipeline
    python examples/run_streaming.py --sim --bar-seconds 60

    # Tick bars instead of time bars (new bar every 500 ticks)
    python examples/run_streaming.py --ib-paper --tick-bars 500
"""

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.streaming.engine import StreamingEngine
from backtester.streaming.tick_handler import Tick, TICK_TRADE
from backtester.broker.paper_broker import PaperBroker
from backtester.costs import ES_FUTURES
from backtester.strategies.es_scalper import ESScalper
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import MeanReversion
from backtester.strategies.ensemble import EnsembleStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_strategy(name: str):
    if name == "scalper":
        return ESScalper(
            fast_ema=5, slow_ema=13, rsi_period=6,
            atr_stop_mult=1.0, atr_target_mult=2.5,
            vwap_dev_entry=0.8, max_hold_bars=60, cooldown_bars=2,
        )
    elif name == "ensemble":
        e = EnsembleStrategy(threshold=0.3)
        e.add_strategy(ESScalper(fast_ema=5, slow_ema=13, atr_stop_mult=1.0, atr_target_mult=2.5), 1.0)
        e.add_strategy(MeanReversion(rsi_period=10, bb_period=15, atr_stop_multiplier=1.5), 0.8)
        return e
    elif name == "sma":
        return SMACrossover(fast_period=8, slow_period=21, atr_stop_multiplier=1.5)
    else:
        return ESScalper()


def run_simulation(engine: StreamingEngine, duration_minutes: int = 5):
    """Simulate tick stream for testing without IB.

    Generates realistic tick data and feeds it through the full pipeline.
    """
    import numpy as np

    logger.info(f"Starting {duration_minutes}-minute tick simulation...")

    rng = np.random.RandomState(42)
    price = 5950.0
    vol = 0.00015  # Per-tick volatility

    # ~4 ticks per second during active trading
    ticks_per_second = 4
    total_ticks = duration_minutes * 60 * ticks_per_second
    tick_interval = 1.0 / ticks_per_second

    engine.broker.connect()
    engine._running = True

    start = time.time()
    # Use simulated timestamps spaced 0.25s apart (4 ticks/sec)
    sim_start = time.time()

    for i in range(total_ticks):
        if not engine._running:
            break

        # Random walk with slight drift and vol clustering
        ret = 0.00001 + vol * rng.randn()
        vol = max(0.00005, min(0.0005, 0.95 * vol + 0.05 * abs(ret)))
        price *= (1 + ret)
        price = round(price / 0.25) * 0.25  # Tick align

        side = 1 if ret > 0 else -1
        size = rng.randint(1, 10)

        # Use properly spaced simulated timestamps so bars complete
        sim_ts = sim_start + i * tick_interval
        engine.inject_tick(price=price, size=size, side=side, timestamp=sim_ts)

        # Print status every 1000 ticks
        if (i + 1) % 1000 == 0:
            status = engine.get_status()
            elapsed = time.time() - start
            tps = (i + 1) / elapsed
            logger.info(
                f"Tick {i+1}/{total_ticks} | Price: {price:.2f} | "
                f"Bars: {status['bars_completed']} | "
                f"Pos: {status['position']} | "
                f"P&L: ${status['daily_pnl']:+,.2f} | "
                f"{tps:.0f} ticks/sec"
            )

    engine._shutdown()


def main():
    parser = argparse.ArgumentParser(description="ES Tick-Streaming Trading Engine")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ib-paper", action="store_true", help="IB paper trading")
    mode.add_argument("--ib-live", action="store_true", help="IB live (REAL MONEY)")
    mode.add_argument("--sim", action="store_true", help="Simulate ticks locally")

    parser.add_argument("--strategy", default="scalper",
                       choices=["scalper", "ensemble", "sma"])
    parser.add_argument("--bar-seconds", type=int, default=300,
                       help="Time bar size in seconds (60=1min, 300=5min)")
    parser.add_argument("--tick-bars", type=int, default=0,
                       help="Tick bar size (e.g., 500). Overrides --bar-seconds")
    parser.add_argument("--capital", type=float, default=50000)
    parser.add_argument("--max-position", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=3.0)
    parser.add_argument("--max-drawdown", type=float, default=20.0)

    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=None)
    parser.add_argument("--sim-minutes", type=int, default=60,
                       help="Minutes to simulate (--sim mode only)")

    args = parser.parse_args()

    # Safety check
    if args.ib_live:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING - REAL MONEY")
        print("!" * 60)
        confirm = input("Type 'I UNDERSTAND' to continue: ")
        if confirm != "I UNDERSTAND":
            sys.exit(0)

    # Build broker
    if args.sim:
        broker = PaperBroker(initial_capital=args.capital, cost_model=ES_FUTURES)
    else:
        from backtester.broker.ib_broker import IBBroker
        port = args.ib_port or (7497 if args.ib_paper else 7496)
        broker = IBBroker(host=args.ib_host, port=port)

    # Build strategy
    strategy = build_strategy(args.strategy)

    # Build engine
    bar_seconds = 0 if args.tick_bars > 0 else args.bar_seconds
    engine = StreamingEngine(
        strategy=strategy,
        broker=broker,
        bar_seconds=bar_seconds,
        tick_bar_size=args.tick_bars,
        max_position=args.max_position,
        initial_capital=args.capital,
        daily_loss_limit_pct=args.daily_loss_limit,
        max_drawdown_pct=args.max_drawdown,
    )

    if args.sim:
        run_simulation(engine, duration_minutes=args.sim_minutes)
    else:
        engine.start()


if __name__ == "__main__":
    main()
