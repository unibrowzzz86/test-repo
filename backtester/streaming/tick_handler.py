"""Tick data handler.

Ingests raw tick-by-tick market data, stores it in a ring buffer,
and dispatches tick events to subscribers. Handles thousands of
ticks per second without allocation pressure.

A Tick is the atomic unit: one trade or quote at a specific
price/size/timestamp. Everything else (bars, signals, orders)
is derived from ticks.
"""

import time
import logging
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Tick:
    """Single market tick (trade or quote)."""
    timestamp: float          # Unix timestamp (time.time()) for speed
    price: float              # Trade price
    size: int                 # Trade size (contracts)
    side: int                 # 1 = buy (uptick), -1 = sell (downtick), 0 = unknown
    tick_type: int            # 0 = trade, 1 = bid, 2 = ask

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)


# Pre-allocated constants
TICK_TRADE = 0
TICK_BID = 1
TICK_ASK = 2
SIDE_BUY = 1
SIDE_SELL = -1
SIDE_UNKNOWN = 0


class TickHandler:
    """High-performance tick data handler.

    Stores ticks in a fixed-size ring buffer to avoid unbounded
    memory growth. Dispatches tick events to registered callbacks.

    Usage:
        handler = TickHandler(buffer_size=100_000)
        handler.on_tick(my_callback)    # Register listener
        handler.process(tick)            # Feed ticks in
    """

    def __init__(self, buffer_size: int = 100_000):
        self._buffer: deque[Tick] = deque(maxlen=buffer_size)
        self._callbacks: list[Callable[[Tick], None]] = []
        self._tick_count = 0
        self._last_price = 0.0
        self._last_bid = 0.0
        self._last_ask = 0.0
        self._session_high = 0.0
        self._session_low = float('inf')
        self._session_volume = 0
        self._session_start: float | None = None

    def on_tick(self, callback: Callable[[Tick], None]):
        """Register a callback for each incoming tick."""
        self._callbacks.append(callback)

    def process(self, tick: Tick):
        """Process a single tick. Called for every incoming tick."""
        self._buffer.append(tick)
        self._tick_count += 1

        if tick.tick_type == TICK_TRADE:
            self._last_price = tick.price
            self._session_volume += tick.size

            if self._session_start is None:
                self._session_start = tick.timestamp
                self._session_high = tick.price
                self._session_low = tick.price
            else:
                if tick.price > self._session_high:
                    self._session_high = tick.price
                if tick.price < self._session_low:
                    self._session_low = tick.price

        elif tick.tick_type == TICK_BID:
            self._last_bid = tick.price
        elif tick.tick_type == TICK_ASK:
            self._last_ask = tick.price

        # Dispatch to all listeners
        for cb in self._callbacks:
            cb(tick)

    def process_batch(self, ticks: list[Tick]):
        """Process a batch of ticks. More efficient than one at a time."""
        for tick in ticks:
            self._buffer.append(tick)
            self._tick_count += 1

            if tick.tick_type == TICK_TRADE:
                self._last_price = tick.price
                self._session_volume += tick.size
                if tick.price > self._session_high:
                    self._session_high = tick.price
                if tick.price < self._session_low:
                    self._session_low = tick.price

        # Dispatch only the last tick to callbacks (for bar building,
        # the aggregator handles each tick individually via on_tick)
        if ticks:
            for cb in self._callbacks:
                for tick in ticks:
                    cb(tick)

    def reset_session(self):
        """Reset session stats (call at start of each trading day)."""
        self._session_high = 0.0
        self._session_low = float('inf')
        self._session_volume = 0
        self._session_start = None
        self._tick_count = 0

    @property
    def last_price(self) -> float:
        return self._last_price

    @property
    def last_bid(self) -> float:
        return self._last_bid

    @property
    def last_ask(self) -> float:
        return self._last_ask

    @property
    def spread(self) -> float:
        if self._last_bid > 0 and self._last_ask > 0:
            return self._last_ask - self._last_bid
        return 0.0

    @property
    def mid_price(self) -> float:
        if self._last_bid > 0 and self._last_ask > 0:
            return (self._last_bid + self._last_ask) / 2
        return self._last_price

    @property
    def session_high(self) -> float:
        return self._session_high

    @property
    def session_low(self) -> float:
        return self._session_low if self._session_low != float('inf') else 0.0

    @property
    def session_volume(self) -> int:
        return self._session_volume

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def get_recent_ticks(self, n: int = 100) -> list[Tick]:
        """Get the last N ticks from the buffer."""
        buf = self._buffer
        start = max(0, len(buf) - n)
        return list(buf)[start:]

    def get_recent_trades(self, n: int = 100) -> list[Tick]:
        """Get the last N trade ticks (excluding quotes)."""
        trades = [t for t in self._buffer if t.tick_type == TICK_TRADE]
        return trades[-n:]

    def get_vwap(self, n_ticks: int = 0) -> float:
        """Calculate VWAP from recent trade ticks."""
        trades = [t for t in self._buffer if t.tick_type == TICK_TRADE]
        if n_ticks > 0:
            trades = trades[-n_ticks:]
        if not trades:
            return 0.0
        total_value = sum(t.price * t.size for t in trades)
        total_volume = sum(t.size for t in trades)
        return total_value / total_volume if total_volume > 0 else 0.0

    def get_tick_imbalance(self, n_ticks: int = 100) -> float:
        """Calculate buy/sell tick imbalance (-1.0 to 1.0).

        Positive = more buy pressure, negative = more sell pressure.
        This is a key signal for tick-level strategies.
        """
        trades = self.get_recent_trades(n_ticks)
        if not trades:
            return 0.0
        buy_vol = sum(t.size for t in trades if t.side == SIDE_BUY)
        sell_vol = sum(t.size for t in trades if t.side == SIDE_SELL)
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total
