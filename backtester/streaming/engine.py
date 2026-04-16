"""Event-driven streaming engine.

The core reactive pipeline:
    tick stream -> TickHandler -> BarAggregator -> Strategy -> Broker

Everything is callback-driven. No polling, no sleep loops.
When a tick arrives, it flows through the entire pipeline
synchronously within microseconds.

Usage:
    engine = StreamingEngine(
        strategy=ESScalper(),
        broker=IBBroker(port=7497),
        bar_seconds=300,  # 5-min bars
    )
    engine.start()  # Connects to IB and starts processing ticks
"""

import logging
import time
import json
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import pandas as pd

from backtester.streaming.tick_handler import Tick, TickHandler, TICK_TRADE
from backtester.streaming.bar_aggregator import Bar, BarAggregator
from backtester.strategies.base import Strategy
from backtester.broker.base import Broker, Order, OrderSide, OrderType
from backtester.costs import CostModel, ES_FUTURES
from backtester.position_sizing import PositionSizer
from backtester.monitoring.monitor import LiveMonitor

logger = logging.getLogger(__name__)


class StreamingEngine:
    """Event-driven live trading engine.

    This is the production execution engine. It:
    1. Streams tick-by-tick data from IB
    2. Aggregates ticks into bars in real-time
    3. Runs your strategy on each completed bar
    4. Submits orders through the broker
    5. Manages risk (kill rules, position limits)
    6. Logs every tick, bar, signal, and order

    The entire pipeline is synchronous and callback-driven:
    no polling threads, no sleep loops, no race conditions.
    """

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        symbol: str = "ES",
        bar_seconds: int = 300,
        tick_bar_size: int = 0,
        point_value: float = 50.0,
        max_position: int = 1,
        initial_capital: float = 50000.0,
        daily_loss_limit_pct: float = 3.0,
        max_drawdown_pct: float = 20.0,
        flatten_eod: bool = True,
        trading_start_hour: int = 9,
        trading_start_minute: int = 30,
        trading_end_hour: int = 16,
        trading_end_minute: int = 0,
        log_dir: str = "logs",
    ):
        self.strategy = strategy
        self.broker = broker
        self.symbol = symbol
        self.point_value = point_value
        self.max_position = max_position
        self.initial_capital = initial_capital
        self.flatten_eod = flatten_eod
        self.trading_start = trading_start_hour * 60 + trading_start_minute
        self.trading_end = trading_end_hour * 60 + trading_end_minute
        self.log_dir = Path(log_dir)

        # Streaming components
        self.tick_handler = TickHandler(buffer_size=500_000)
        self.bar_aggregator = BarAggregator(
            bar_seconds=bar_seconds,
            tick_bar_size=tick_bar_size,
            history_size=500,
        )

        # Risk management
        self.position_sizer = PositionSizer(
            max_position_size=max_position,
            daily_loss_limit_pct=daily_loss_limit_pct,
            drawdown_stop_pct=max_drawdown_pct,
        )
        self.monitor = LiveMonitor(
            max_drawdown_kill_pct=max_drawdown_pct,
            max_drawdown_alert_pct=max_drawdown_pct * 0.5,
        )

        # State
        self._position = 0
        self._entry_price = 0.0
        self._daily_pnl = 0.0
        self._peak_equity = initial_capital
        self._today: str | None = None
        self._running = False
        self._last_signal = 0
        self._trade_log: list[dict] = []
        self._lock = Lock()

        # Wire up the pipeline:
        # ticks -> tick_handler -> bar_aggregator -> _on_bar_complete
        self.tick_handler.on_tick(self.bar_aggregator.process_tick)
        self.bar_aggregator.on_bar(self._on_bar_complete)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        self.log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        handler = logging.FileHandler(self.log_dir / f"streaming_{ts}.log")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # ═══════════════════════════════════════════
    # CORE EVENT HANDLER: called on every completed bar
    # ═══════════════════════════════════════════

    def _on_bar_complete(self, bar: Bar):
        """Called when a bar completes. This is where signals happen.

        This is the heart of the engine. On every completed bar:
        1. Check if we're in trading hours
        2. Check kill rules
        3. Run strategy to get signal
        4. Compare signal to current position
        5. Submit orders if needed
        """
        with self._lock:
            now = datetime.fromtimestamp(bar.timestamp)

            # Daily reset
            today = now.strftime("%Y-%m-%d")
            if self._today != today:
                if self._today is not None:
                    logger.info(f"Day change. Yesterday P&L: ${self._daily_pnl:,.2f}")
                self._daily_pnl = 0.0
                self._today = today
                self.tick_handler.reset_session()

            # Log bar
            logger.info(
                f"BAR | O:{bar.open:.2f} H:{bar.high:.2f} L:{bar.low:.2f} "
                f"C:{bar.close:.2f} V:{bar.volume} Δ:{bar.delta:+d} "
                f"Ticks:{bar.tick_count} VWAP:{bar.vwap:.2f}"
            )

            # Check trading hours
            current_min = now.hour * 60 + now.minute
            if current_min < self.trading_start or current_min >= self.trading_end:
                if self.flatten_eod and self._position != 0:
                    logger.info("EOD: Flattening positions")
                    self._close_position(bar.close, "eod_flatten")
                return

            if now.weekday() >= 5:
                return

            # Check kill rules
            equity = self.initial_capital + self._daily_pnl
            if self._position != 0:
                unrealized = (bar.close - self._entry_price) * self._position * self.point_value
                equity += unrealized

            self._peak_equity = max(self._peak_equity, equity)
            dd_pct = (self._peak_equity - equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0
            daily_pnl_pct = (self._daily_pnl / self.initial_capital) * 100

            self.monitor.update_equity(now, equity)

            if self.position_sizer.should_stop(dd_pct, daily_pnl_pct):
                if self._position != 0:
                    logger.critical(
                        f"KILL RULE: DD={dd_pct:.1f}% daily={daily_pnl_pct:.1f}%. "
                        f"Flattening."
                    )
                    self._close_position(bar.close, "kill_rule")
                return

            # Need enough bars for strategy warmup
            if not self.bar_aggregator.has_enough_history:
                return

            # Run strategy
            df = self.bar_aggregator.to_dataframe()
            signals = self.strategy.generate_signals(df)
            new_signal = int(signals.iloc[-1])

            if new_signal != self._position:
                logger.info(
                    f"SIGNAL: {self._position} -> {new_signal} @ {bar.close:.2f}"
                )
                self._execute_signal_change(new_signal, bar.close)

            self._last_signal = new_signal

    # ═══════════════════════════════════════════
    # ORDER EXECUTION
    # ═══════════════════════════════════════════

    def _execute_signal_change(self, target: int, price: float):
        """Execute a position change from current to target."""
        # Close existing position
        if self._position != 0:
            self._close_position(price, "signal")

        # Open new position
        if target != 0:
            self._open_position(target, price)

    def _open_position(self, direction: int, price: float):
        """Open a new position."""
        side = OrderSide.BUY if direction > 0 else OrderSide.SELL
        order = Order(
            symbol=self.symbol,
            side=side,
            quantity=abs(direction),
            order_type=OrderType.MARKET,
        )
        order_id = self.broker.submit_order(order)
        self._position = direction
        self._entry_price = price

        logger.info(
            f"OPEN {'LONG' if direction > 0 else 'SHORT'} {abs(direction)} "
            f"@ {price:.2f} [order: {order_id}]"
        )

    def _close_position(self, price: float, reason: str):
        """Close current position and record P&L."""
        pnl = (price - self._entry_price) * self._position * self.point_value
        self._daily_pnl += pnl

        side = OrderSide.SELL if self._position > 0 else OrderSide.BUY
        order = Order(
            symbol=self.symbol,
            side=side,
            quantity=abs(self._position),
            order_type=OrderType.MARKET,
        )
        order_id = self.broker.submit_order(order)

        trade = {
            "timestamp": datetime.now().isoformat(),
            "direction": "LONG" if self._position > 0 else "SHORT",
            "entry": self._entry_price,
            "exit": price,
            "pnl": round(pnl, 2),
            "reason": reason,
            "daily_pnl": round(self._daily_pnl, 2),
        }
        self._trade_log.append(trade)
        self.monitor.record_trade({"pnl": pnl, "timestamp": datetime.now()})

        logger.info(
            f"CLOSE {'LONG' if self._position > 0 else 'SHORT'} "
            f"@ {price:.2f} | P&L: ${pnl:+,.2f} | Daily: ${self._daily_pnl:+,.2f} "
            f"| Reason: {reason} [order: {order_id}]"
        )

        self._position = 0
        self._entry_price = 0.0

    # ═══════════════════════════════════════════
    # IB TICK STREAMING
    # ═══════════════════════════════════════════

    def start(self):
        """Start the streaming engine.

        Connects to IB, subscribes to tick data, and processes
        ticks through the pipeline. Blocks until shutdown.
        """
        logger.info("=" * 60)
        logger.info("STREAMING ENGINE STARTING")
        logger.info(f"  Strategy: {self.strategy.name}")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  Bar size: {self.bar_aggregator.bar_seconds}s")
        logger.info(f"  Max position: {self.max_position}")
        logger.info("=" * 60)

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            return

        self._running = True

        try:
            self._subscribe_ticks()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _subscribe_ticks(self):
        """Subscribe to IB tick-by-tick data and process."""
        try:
            from ib_insync import Future
        except ImportError:
            raise ImportError("ib_insync required. Install: pip install ib_insync")

        if not hasattr(self.broker, 'ib') or self.broker.ib is None:
            raise ConnectionError("Broker must be IBBroker with active connection")

        ib = self.broker.ib
        contract = Future(symbol="ES", exchange="CME", currency="USD")
        ib.qualifyContracts(contract)

        # Subscribe to tick-by-tick Last (trade) data
        ticker = ib.reqTickByTickData(contract, "AllLast")

        logger.info(f"Subscribed to tick stream for {self.symbol}")

        # IB's event loop: ib.sleep() processes events and calls our callbacks
        def on_tick_update(ticker, tick_data):
            """Called by IB for each tick."""
            if tick_data is None:
                return

            t = Tick(
                timestamp=tick_data.time.timestamp() if hasattr(tick_data, 'time') else time.time(),
                price=tick_data.price if hasattr(tick_data, 'price') else 0,
                size=int(tick_data.size) if hasattr(tick_data, 'size') else 1,
                side=1 if hasattr(tick_data, 'exchange') and tick_data.price >= self.tick_handler.last_price else -1,
                tick_type=0,  # Trade
            )
            self.tick_handler.process(t)

        ticker.updateEvent += on_tick_update

        # Also get bid/ask quotes for spread monitoring
        ib.reqMktData(contract, "", False, False)

        # Block in IB event loop
        while self._running:
            ib.sleep(0.01)  # Process IB events every 10ms

    def _shutdown_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def _shutdown(self):
        """Clean shutdown: flatten, disconnect, save logs."""
        if self._position != 0:
            logger.info("Shutdown: flattening positions")
            price = self.tick_handler.last_price
            if price > 0:
                self._close_position(price, "shutdown")

        self._save_trade_log()

        if self.broker.is_connected():
            self.broker.disconnect()

        status = self.monitor.get_status()
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Trades: {status.get('total_trades', 0)}")
        logger.info(f"  P&L: ${status.get('total_pnl', 0):+,.2f}")
        logger.info(f"  Win Rate: {status.get('win_rate', 0):.1f}%")
        logger.info(f"  Ticks processed: {self.tick_handler.tick_count:,}")
        logger.info(f"  Bars completed: {self.bar_aggregator.bar_count}")
        logger.info("=" * 60)

    def _save_trade_log(self):
        self.log_dir.mkdir(exist_ok=True)
        f = self.log_dir / f"trades_{self._today or 'session'}.json"
        with open(f, "w") as fh:
            json.dump(self._trade_log, fh, indent=2)

    # ═══════════════════════════════════════════
    # MANUAL TICK INJECTION (for paper trading / replay)
    # ═══════════════════════════════════════════

    def inject_tick(
        self, price: float, size: int = 1, side: int = 0,
        timestamp: float | None = None,
    ):
        """Manually inject a tick (for paper trading or tick replay).

        Use this when not connected to IB — feed ticks from any source:
        - Recorded tick data files
        - Another data provider's WebSocket
        - Simulated ticks for testing

        The tick flows through the full pipeline:
        tick -> handler -> aggregator -> strategy -> broker

        Args:
            price: Trade price.
            size: Trade size (contracts).
            side: 1=buy, -1=sell, 0=unknown.
            timestamp: Unix timestamp. None = now (wall clock).
        """
        tick = Tick(
            timestamp=timestamp if timestamp is not None else time.time(),
            price=price,
            size=size,
            side=side,
            tick_type=TICK_TRADE,
        )
        self.tick_handler.process(tick)

        # Update paper broker price if applicable
        if hasattr(self.broker, 'set_price'):
            self.broker.set_price(self.symbol, price)

    def replay_ticks(self, ticks: list[dict], speed: float = 1.0):
        """Replay recorded tick data through the engine.

        Args:
            ticks: List of dicts with keys: timestamp, price, size, side
            speed: Playback speed multiplier (1.0 = real-time, 0 = instant)
        """
        logger.info(f"Replaying {len(ticks)} ticks (speed={speed}x)")

        if not ticks:
            return

        start_time = ticks[0].get("timestamp", time.time())
        replay_start = time.time()

        for tick_data in ticks:
            if not self._running:
                break

            # Timing
            if speed > 0:
                tick_offset = tick_data.get("timestamp", start_time) - start_time
                target_time = replay_start + tick_offset / speed
                wait = target_time - time.time()
                if wait > 0:
                    time.sleep(wait)

            self.inject_tick(
                price=tick_data["price"],
                size=tick_data.get("size", 1),
                side=tick_data.get("side", 0),
            )

        logger.info(f"Replay complete. {self.bar_aggregator.bar_count} bars built.")

    def get_status(self) -> dict:
        """Get current engine status."""
        return {
            "running": self._running,
            "position": self._position,
            "entry_price": self._entry_price,
            "last_price": self.tick_handler.last_price,
            "spread": self.tick_handler.spread,
            "daily_pnl": round(self._daily_pnl, 2),
            "session_high": self.tick_handler.session_high,
            "session_low": self.tick_handler.session_low,
            "session_volume": self.tick_handler.session_volume,
            "ticks_processed": self.tick_handler.tick_count,
            "bars_completed": self.bar_aggregator.bar_count,
            "last_signal": self._last_signal,
            "trades_today": len(self._trade_log),
            "monitor": self.monitor.get_status(),
        }
