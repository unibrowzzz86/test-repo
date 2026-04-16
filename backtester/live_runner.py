"""Live trading runner.

The main execution loop that:
1. Pulls real-time data (from IB or data provider)
2. Generates signals from your strategy
3. Manages orders through the broker
4. Monitors performance and enforces kill rules
5. Logs everything

Supports two modes:
- Autonomous: Runs your strategy signals directly
- Webhook: Receives signals from TradingView

Usage:
    # Autonomous mode with paper broker
    python -m backtester.live_runner --mode paper --strategy ensemble

    # Autonomous mode with IB
    python -m backtester.live_runner --mode ib --port 7497

    # TradingView webhook mode
    python -m backtester.live_runner --mode webhook --webhook-port 5000
"""

import logging
import time
import json
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field

from backtester.broker.base import Broker, Order, OrderSide, OrderType
from backtester.broker.paper_broker import PaperBroker
from backtester.strategies.base import Strategy
from backtester.costs import ES_FUTURES
from backtester.position_sizing import PositionSizer
from backtester.monitoring.monitor import LiveMonitor

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    """Configuration for live trading."""
    symbol: str = "ES"
    bar_interval_seconds: int = 1800  # 30 minutes
    point_value: float = 50.0
    tick_size: float = 0.25
    max_position: int = 1
    initial_capital: float = 50000.0

    # Risk limits
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 20.0
    max_consecutive_losses: int = 8

    # Trading hours (ET)
    trading_start_hour: int = 9
    trading_start_minute: int = 30
    trading_end_hour: int = 16
    trading_end_minute: int = 0
    flatten_eod: bool = True  # Close all positions at end of day

    # Logging
    log_dir: str = "logs"
    log_trades: bool = True


class LiveRunner:
    """Main live trading execution loop."""

    def __init__(
        self,
        broker: Broker,
        strategy: Strategy | None = None,
        config: LiveConfig | None = None,
        monitor: LiveMonitor | None = None,
        position_sizer: PositionSizer | None = None,
    ):
        self.broker = broker
        self.strategy = strategy
        self.config = config or LiveConfig()
        self.position_sizer = position_sizer or PositionSizer(
            max_position_size=self.config.max_position,
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            drawdown_stop_pct=self.config.max_drawdown_pct,
        )
        self.monitor = monitor or LiveMonitor(
            max_drawdown_kill_pct=self.config.max_drawdown_pct,
            max_consecutive_losses=self.config.max_consecutive_losses,
        )

        self._running = False
        self._bars: list[dict] = []
        self._current_position = 0
        self._entry_price = 0.0
        self._daily_pnl = 0.0
        self._today = None
        self._peak_equity = self.config.initial_capital
        self._trade_log: list[dict] = []

        # Setup logging
        self._setup_logging()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _setup_logging(self):
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"live_trading_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    def _shutdown_handler(self, signum, frame):
        logger.info("Shutdown signal received, flattening positions...")
        self._running = False
        self._flatten_all()
        self._save_trade_log()
        logger.info("Shutdown complete")
        sys.exit(0)

    def run(self):
        """Start the live trading loop.

        This is the main event loop. It:
        1. Connects to the broker
        2. Waits for market open
        3. On each bar interval:
           a. Gets latest price/bar data
           b. Runs strategy to generate signal
           c. Compares signal to current position
           d. Submits orders if position change needed
           e. Checks kill rules
           f. Logs everything
        4. Flattens at EOD if configured
        """
        logger.info("=" * 60)
        logger.info("LIVE TRADING RUNNER STARTING")
        logger.info(f"  Strategy: {self.strategy.name if self.strategy else 'Webhook Mode'}")
        logger.info(f"  Symbol: {self.config.symbol}")
        logger.info(f"  Bar interval: {self.config.bar_interval_seconds}s")
        logger.info(f"  Max position: {self.config.max_position}")
        logger.info(f"  Daily loss limit: {self.config.daily_loss_limit_pct}%")
        logger.info(f"  Max drawdown: {self.config.max_drawdown_pct}%")
        logger.info("=" * 60)

        if not self.broker.connect():
            logger.error("Failed to connect to broker. Exiting.")
            return

        self._running = True
        last_bar_time = None

        while self._running:
            try:
                now = datetime.now()

                # Reset daily P&L on new day
                today = now.date()
                if self._today != today:
                    if self._today is not None:
                        logger.info(f"New day. Previous day P&L: ${self._daily_pnl:,.2f}")
                    self._daily_pnl = 0.0
                    self._today = today

                # Check if within trading hours
                if not self._is_trading_hours(now):
                    if self.config.flatten_eod and self._current_position != 0:
                        logger.info("End of trading hours - flattening positions")
                        self._flatten_all()
                    time.sleep(10)
                    continue

                # Check kill rules
                equity = self.broker.get_account_value()
                if equity > 0:
                    self._peak_equity = max(self._peak_equity, equity)
                    dd_pct = (self._peak_equity - equity) / self._peak_equity * 100
                    daily_pnl_pct = (self._daily_pnl / self.config.initial_capital) * 100

                    if self.position_sizer.should_stop(dd_pct, daily_pnl_pct):
                        if self._current_position != 0:
                            logger.critical(
                                f"KILL RULE TRIGGERED: DD={dd_pct:.1f}%, "
                                f"Daily P&L={daily_pnl_pct:.1f}%. Flattening."
                            )
                            self._flatten_all()
                        time.sleep(60)
                        continue

                    self.monitor.update_equity(now, equity)

                # Strategy signal generation (autonomous mode)
                if self.strategy:
                    # Check if it's time for a new bar
                    if last_bar_time is None or \
                       (now - last_bar_time).total_seconds() >= self.config.bar_interval_seconds:

                        self._process_strategy_signal()
                        last_bar_time = now

                # Sleep until next check
                time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

        self._shutdown()

    def _process_strategy_signal(self):
        """Run strategy on accumulated bars and act on signal."""
        if not self.strategy:
            return

        # Get latest price from broker
        if hasattr(self.broker, 'get_real_time_price'):
            price = self.broker.get_real_time_price(self.config.symbol)
        elif hasattr(self.broker, '_current_prices'):
            price = self.broker._current_prices.get(self.config.symbol, 0)
        else:
            return

        if not price:
            return

        # Add bar to history
        now = datetime.now()
        self._bars.append({
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100000,
        })

        # Need enough bars for strategy warmup
        if len(self._bars) < 60:
            return

        # Build DataFrame for strategy
        bar_data = pd.DataFrame(self._bars[-200:])  # Keep last 200 bars
        bar_data.index = pd.date_range(end=now, periods=len(bar_data), freq="30min")

        # Generate signal
        signals = self.strategy.generate_signals(bar_data)
        signal = int(signals.iloc[-1])

        if signal != self._current_position:
            self._execute_signal(signal, price)

    def _execute_signal(self, target_position: int, price: float):
        """Execute a position change."""
        current = self._current_position

        if current != 0:
            # Close current position
            side = OrderSide.SELL if current > 0 else OrderSide.BUY
            order = Order(
                symbol=self.config.symbol,
                side=side,
                quantity=abs(current),
                order_type=OrderType.MARKET,
            )
            self.broker.submit_order(order)

            # Calculate P&L
            pnl = (price - self._entry_price) * current * self.config.point_value
            self._daily_pnl += pnl
            self._log_trade(current, self._entry_price, price, pnl)

            logger.info(
                f"CLOSED: {'LONG' if current > 0 else 'SHORT'} {abs(current)} "
                f"@ {price:.2f} | P&L: ${pnl:,.2f} | Daily: ${self._daily_pnl:,.2f}"
            )

        if target_position != 0:
            # Open new position
            side = OrderSide.BUY if target_position > 0 else OrderSide.SELL
            order = Order(
                symbol=self.config.symbol,
                side=side,
                quantity=abs(target_position),
                order_type=OrderType.MARKET,
            )
            self.broker.submit_order(order)
            self._entry_price = price
            logger.info(
                f"OPENED: {'LONG' if target_position > 0 else 'SHORT'} "
                f"{abs(target_position)} @ {price:.2f}"
            )

        self._current_position = target_position

    def _flatten_all(self):
        """Close all positions."""
        if self._current_position != 0:
            pos = self.broker.get_position(self.config.symbol)
            price = 0
            if hasattr(self.broker, '_current_prices'):
                price = self.broker._current_prices.get(self.config.symbol, 0)

            self.broker.flatten(self.config.symbol)

            if price > 0:
                pnl = (price - self._entry_price) * self._current_position * self.config.point_value
                self._daily_pnl += pnl
                self._log_trade(self._current_position, self._entry_price, price, pnl)
                logger.info(f"FLATTEN: P&L ${pnl:,.2f}")

            self._current_position = 0

    def _is_trading_hours(self, now: datetime) -> bool:
        """Check if currently within trading hours."""
        current_minutes = now.hour * 60 + now.minute
        start_minutes = self.config.trading_start_hour * 60 + self.config.trading_start_minute
        end_minutes = self.config.trading_end_hour * 60 + self.config.trading_end_minute

        # Weekday check
        if now.weekday() >= 5:  # Saturday/Sunday
            return False

        return start_minutes <= current_minutes < end_minutes

    def _log_trade(self, direction: int, entry: float, exit_price: float, pnl: float):
        """Log a completed trade."""
        trade = {
            "timestamp": datetime.now().isoformat(),
            "direction": "LONG" if direction > 0 else "SHORT",
            "entry_price": entry,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "daily_pnl": round(self._daily_pnl, 2),
        }
        self._trade_log.append(trade)

        self.monitor.record_trade({"pnl": pnl, "timestamp": datetime.now()})

        if self.config.log_trades:
            self._save_trade_log()

    def _save_trade_log(self):
        """Save trade log to disk."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"trades_{self._today or 'unknown'}.json"
        with open(log_file, "w") as f:
            json.dump(self._trade_log, f, indent=2)

    def _shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self._flatten_all()
        self._save_trade_log()
        self.broker.disconnect()

        # Print summary
        status = self.monitor.get_status()
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Total trades: {status.get('total_trades', 0)}")
        logger.info(f"  Total P&L: ${status.get('total_pnl', 0):,.2f}")
        logger.info(f"  Win rate: {status.get('win_rate', 0):.1f}%")
        logger.info(f"  Alerts: {status.get('num_alerts', 0)}")
        logger.info("=" * 60)

    def get_status(self) -> dict:
        """Get current running status."""
        equity = self.broker.get_account_value() if self.broker.is_connected() else 0
        return {
            "running": self._running,
            "connected": self.broker.is_connected(),
            "current_position": self._current_position,
            "entry_price": self._entry_price,
            "daily_pnl": round(self._daily_pnl, 2),
            "equity": round(equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "bars_collected": len(self._bars),
            "trades_today": len([t for t in self._trade_log
                                if t.get("timestamp", "").startswith(str(self._today))]),
            "monitor": self.monitor.get_status(),
        }


def main():
    """CLI entry point for live runner."""
    import argparse

    parser = argparse.ArgumentParser(description="ES Futures Live Trading Runner")
    parser.add_argument("--mode", choices=["paper", "ib", "webhook"], default="paper",
                       help="Trading mode: paper, ib (Interactive Brokers), or webhook (TradingView)")
    parser.add_argument("--strategy", default="ensemble",
                       help="Strategy: ensemble, sma, mean_reversion, momentum, vwap")
    parser.add_argument("--capital", type=float, default=50000,
                       help="Initial capital")
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=7497,
                       help="IB port: 7497=paper, 7496=live")
    parser.add_argument("--webhook-port", type=int, default=5000)
    parser.add_argument("--webhook-secret", default="")
    parser.add_argument("--max-position", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=3.0)
    parser.add_argument("--max-drawdown", type=float, default=20.0)
    parser.add_argument("--flatten-eod", action="store_true", default=True)

    args = parser.parse_args()

    # Setup broker
    if args.mode == "paper":
        broker = PaperBroker(initial_capital=args.capital)
    elif args.mode == "ib":
        from backtester.broker.ib_broker import IBBroker
        broker = IBBroker(host=args.ib_host, port=args.ib_port)
    elif args.mode == "webhook":
        broker = PaperBroker(initial_capital=args.capital)  # Or IBBroker
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Setup strategy
    strategy = None
    if args.mode != "webhook":
        from backtester.strategies.sma_crossover import SMACrossover
        from backtester.strategies.mean_reversion import MeanReversion
        from backtester.strategies.momentum_breakout import MomentumBreakout
        from backtester.strategies.vwap_strategy import VWAPStrategy
        from backtester.strategies.ensemble import EnsembleStrategy

        if args.strategy == "ensemble":
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
        elif args.strategy == "sma":
            strategy = SMACrossover(fast_period=8, slow_period=21, atr_stop_multiplier=1.5)
        elif args.strategy == "mean_reversion":
            strategy = MeanReversion(rsi_period=10, bb_period=15, atr_stop_multiplier=1.5)
        elif args.strategy == "momentum":
            strategy = MomentumBreakout(lookback_period=15, atr_stop_multiplier=2.0)
        elif args.strategy == "vwap":
            strategy = VWAPStrategy(vwap_dev_threshold=1.2, atr_stop_multiplier=1.5)

    config = LiveConfig(
        max_position=args.max_position,
        initial_capital=args.capital,
        daily_loss_limit_pct=args.daily_loss_limit,
        max_drawdown_pct=args.max_drawdown,
        flatten_eod=args.flatten_eod,
    )

    runner = LiveRunner(broker=broker, strategy=strategy, config=config)

    if args.mode == "webhook":
        # Start webhook receiver in background, then run main loop
        from backtester.broker.tradingview_webhook import TradingViewReceiver
        receiver = TradingViewReceiver(
            broker=broker,
            secret=args.webhook_secret,
            max_position=args.max_position,
        )
        receiver.start_background(port=args.webhook_port)
        logger.info(f"TradingView webhook listening on port {args.webhook_port}")
        # Keep running (webhook handles signals)
        try:
            while True:
                time.sleep(60)
                status = runner.get_status()
                logger.info(f"Status: {json.dumps(status, default=str)}")
        except KeyboardInterrupt:
            runner._shutdown()
    else:
        runner.run()


if __name__ == "__main__":
    main()
