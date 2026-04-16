"""Real-time bar aggregator.

Builds OHLCV bars from a raw tick stream on the fly.
Supports time-based bars (1m, 5m, etc.) and tick-based bars
(e.g., every 500 ticks). Fires a callback when each bar completes,
which is what drives the signal engine.

This is the bridge between raw ticks and your strategies:
    ticks -> BarAggregator -> completed bars -> strategy signals -> orders
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from collections import deque

import pandas as pd
import numpy as np

from backtester.streaming.tick_handler import Tick, TICK_TRADE

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar built from ticks."""
    timestamp: float        # Bar open time (unix)
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int
    vwap: float             # Volume-weighted average price for this bar
    buy_volume: int         # Volume on upticks
    sell_volume: int        # Volume on downticks
    is_complete: bool = False

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def delta(self) -> int:
        """Order flow delta (buy volume - sell volume)."""
        return self.buy_volume - self.sell_volume


class BarAggregator:
    """Builds bars from a tick stream in real-time.

    Supports two modes:
    - Time-based: complete a bar every N seconds (e.g., 60 for 1-min bars)
    - Tick-based: complete a bar every N ticks

    When a bar completes, on_bar callbacks fire with the finished bar.
    The strategy engine hooks into these callbacks.

    Usage:
        agg = BarAggregator(bar_seconds=300)  # 5-minute bars
        agg.on_bar(my_strategy_handler)
        # Feed ticks from tick handler:
        tick_handler.on_tick(agg.process_tick)
    """

    def __init__(
        self,
        bar_seconds: int = 60,
        tick_bar_size: int = 0,
        history_size: int = 500,
    ):
        """
        Args:
            bar_seconds: Seconds per bar (0 to disable time-based bars).
            tick_bar_size: Ticks per bar (0 to disable tick-based bars).
            history_size: Number of completed bars to keep in memory.
        """
        if bar_seconds <= 0 and tick_bar_size <= 0:
            raise ValueError("Must set bar_seconds or tick_bar_size > 0")

        self.bar_seconds = bar_seconds
        self.tick_bar_size = tick_bar_size
        self.use_time_bars = bar_seconds > 0
        self.history_size = history_size

        self._current_bar: Bar | None = None
        self._completed_bars: deque[Bar] = deque(maxlen=history_size)
        self._bar_callbacks: list[Callable[[Bar], None]] = []
        self._bar_count = 0

        # Running VWAP state for current bar
        self._cum_tp_vol = 0.0
        self._cum_vol = 0

    def on_bar(self, callback: Callable[[Bar], None]):
        """Register a callback for each completed bar."""
        self._bar_callbacks.append(callback)

    def process_tick(self, tick: Tick):
        """Process a single tick — called for every incoming tick.

        This is the hot path. Must be fast.
        """
        if tick.tick_type != TICK_TRADE:
            return  # Only build bars from trades

        price = tick.price
        size = tick.size
        ts = tick.timestamp

        if self._current_bar is None:
            # Start a new bar
            self._start_bar(ts, price, size, tick.side)
            return

        bar = self._current_bar

        # Check if bar should close
        should_close = False
        if self.use_time_bars:
            bar_end_time = bar.timestamp + self.bar_seconds
            if ts >= bar_end_time:
                should_close = True
        else:
            if bar.tick_count >= self.tick_bar_size:
                should_close = True

        if should_close:
            # Close current bar
            bar.is_complete = True
            self._completed_bars.append(bar)
            self._bar_count += 1
            self._fire_bar(bar)

            # Handle time gaps: if tick is way past the bar boundary,
            # we may have missed bars (e.g., during a halt). Start fresh.
            self._start_bar(ts, price, size, tick.side)
            return

        # Update current bar with this tick
        if price > bar.high:
            bar.high = price
        if price < bar.low:
            bar.low = price
        bar.close = price
        bar.volume += size
        bar.tick_count += 1

        if tick.side > 0:
            bar.buy_volume += size
        elif tick.side < 0:
            bar.sell_volume += size

        # Update VWAP
        self._cum_tp_vol += price * size
        self._cum_vol += size
        bar.vwap = self._cum_tp_vol / self._cum_vol if self._cum_vol > 0 else price

    def _start_bar(self, ts: float, price: float, size: int, side: int):
        """Start a new bar."""
        # Align timestamp to bar boundary for time bars
        if self.use_time_bars:
            bar_ts = ts - (ts % self.bar_seconds)
        else:
            bar_ts = ts

        self._current_bar = Bar(
            timestamp=bar_ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            tick_count=1,
            vwap=price,
            buy_volume=size if side > 0 else 0,
            sell_volume=size if side < 0 else 0,
        )
        self._cum_tp_vol = price * size
        self._cum_vol = size

    def _fire_bar(self, bar: Bar):
        """Dispatch completed bar to all listeners."""
        for cb in self._bar_callbacks:
            try:
                cb(bar)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}", exc_info=True)

    def get_bar_history(self, n: int = 0) -> list[Bar]:
        """Get completed bar history. n=0 returns all."""
        bars = list(self._completed_bars)
        if n > 0:
            return bars[-n:]
        return bars

    def to_dataframe(self, n: int = 0) -> pd.DataFrame:
        """Convert bar history to a DataFrame for strategy consumption.

        This is what gets passed to strategy.generate_signals().
        """
        bars = self.get_bar_history(n)
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        data = {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
        index = pd.DatetimeIndex([b.datetime for b in bars])
        return pd.DataFrame(data, index=index)

    @property
    def current_bar(self) -> Bar | None:
        return self._current_bar

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def has_enough_history(self) -> bool:
        """True when we have enough bars for strategy warmup."""
        return len(self._completed_bars) >= 60
