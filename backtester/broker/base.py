"""Base broker interface.

Defines the contract that all broker implementations must follow.
Both paper trading and live trading use this same interface,
so switching from paper to live is a one-line change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    fill_price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    symbol: str
    quantity: int  # Positive = long, negative = short
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class Broker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker. Returns True if successful."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker connection is alive."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order. Returns order ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Get current position for a symbol."""
        pass

    @abstractmethod
    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def get_account_value(self) -> float:
        """Get total account value (equity)."""
        pass

    @abstractmethod
    def get_buying_power(self) -> float:
        """Get available buying power."""
        pass

    @abstractmethod
    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        """Get recent fills."""
        pass

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Get all open/pending orders."""
        pass

    def flatten(self, symbol: str) -> str | None:
        """Close any open position for a symbol. Returns order ID or None."""
        pos = self.get_position(symbol)
        if pos is None or pos.quantity == 0:
            return None
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        order = Order(
            symbol=symbol,
            side=side,
            quantity=abs(pos.quantity),
            order_type=OrderType.MARKET,
        )
        return self.submit_order(order)

    def flatten_all(self) -> list[str]:
        """Close all open positions."""
        order_ids = []
        for pos in self.get_all_positions():
            if pos.quantity != 0:
                oid = self.flatten(pos.symbol)
                if oid:
                    order_ids.append(oid)
        return order_ids
