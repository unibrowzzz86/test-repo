"""Paper trading broker.

Simulates order execution with realistic fills, slippage, and commission.
Uses the same Broker interface as IBBroker so switching from paper to
live is a one-line change:

    # Paper trading
    broker = PaperBroker(initial_capital=50000)

    # Go live (just swap this line)
    broker = IBBroker(port=7496)
"""

import logging
from datetime import datetime
from backtester.broker.base import (
    Broker, Order, Fill, Position, OrderSide, OrderType, OrderStatus,
)
from backtester.costs import CostModel, ES_FUTURES

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Simulated broker for paper trading."""

    def __init__(
        self,
        initial_capital: float = 50000.0,
        cost_model: CostModel | None = None,
        point_value: float = 50.0,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.cost_model = cost_model or ES_FUTURES
        self.point_value = point_value

        self._connected = False
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._open_orders: list[Order] = []
        self._next_order_id = 1
        self._current_prices: dict[str, float] = {}

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"Paper broker connected. Capital: ${self.initial_capital:,.2f}")
        return True

    def disconnect(self):
        self._connected = False
        logger.info("Paper broker disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def set_price(self, symbol: str, price: float):
        """Update the current market price (called by the live runner)."""
        self._current_prices[symbol] = price
        # Update unrealized P&L
        if symbol in self._positions:
            pos = self._positions[symbol]
            if pos.quantity != 0:
                pos.unrealized_pnl = (
                    (price - pos.avg_price) * pos.quantity * self.point_value
                )
        # Check stop orders
        self._check_stop_orders(symbol, price)

    def submit_order(self, order: Order) -> str:
        if not self._connected:
            raise ConnectionError("Paper broker not connected")

        order.order_id = f"PAPER_{self._next_order_id}"
        self._next_order_id += 1
        order.status = OrderStatus.SUBMITTED
        order.timestamp = datetime.now()

        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            self._open_orders.append(order)
            logger.info(f"Stop order placed: {order.order_id} @ {order.stop_price}")
        elif order.order_type == OrderType.LIMIT:
            # For simplicity, check if limit is immediately fillable
            price = self._current_prices.get(order.symbol, 0)
            if (order.side == OrderSide.BUY and price <= order.limit_price) or \
               (order.side == OrderSide.SELL and price >= order.limit_price):
                self._execute_at_price(order, order.limit_price)
            else:
                self._open_orders.append(order)

        return order.order_id

    def _execute_market_order(self, order: Order):
        """Fill a market order at current price with slippage."""
        price = self._current_prices.get(order.symbol, 0)
        if price == 0:
            logger.warning(f"No price available for {order.symbol}, order rejected")
            order.status = OrderStatus.REJECTED
            return

        # Apply slippage
        direction = 1 if order.side == OrderSide.BUY else -1
        fill_price = self.cost_model.adjusted_entry_price(price, direction, order.quantity)

        self._execute_at_price(order, fill_price)

    def _execute_at_price(self, order: Order, fill_price: float):
        """Execute an order at a specific price."""
        commission = self.cost_model.commission_per_contract * order.quantity

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            timestamp=datetime.now(),
        )
        self._fills.append(fill)
        order.status = OrderStatus.FILLED

        # Update position
        self._update_position(order.symbol, order.side, order.quantity, fill_price, commission)

        logger.info(
            f"FILL: {order.order_id} {order.side.value} {order.quantity} {order.symbol} "
            f"@ {fill_price:.2f} (commission: ${commission:.2f})"
        )

    def _update_position(
        self, symbol: str, side: OrderSide, quantity: int,
        fill_price: float, commission: float,
    ):
        """Update position and cash after a fill."""
        signed_qty = quantity if side == OrderSide.BUY else -quantity

        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol, quantity=0, avg_price=0,
            )

        pos = self._positions[symbol]
        old_qty = pos.quantity
        new_qty = old_qty + signed_qty

        if old_qty == 0:
            # New position
            pos.avg_price = fill_price
        elif (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
            # Adding to position
            total_cost = pos.avg_price * abs(old_qty) + fill_price * abs(signed_qty)
            pos.avg_price = total_cost / abs(new_qty)
        else:
            # Reducing/closing/reversing position
            closed_qty = min(abs(old_qty), abs(signed_qty))
            realized = (fill_price - pos.avg_price) * closed_qty * (1 if old_qty > 0 else -1)
            realized *= self.point_value
            pos.realized_pnl += realized
            self.cash += realized

            if abs(new_qty) > abs(old_qty):
                # Reversed position
                pos.avg_price = fill_price

        pos.quantity = new_qty
        self.cash -= commission

    def _check_stop_orders(self, symbol: str, price: float):
        """Check if any stop orders should be triggered."""
        triggered = []
        for order in self._open_orders:
            if order.symbol != symbol:
                continue
            if order.order_type == OrderType.STOP:
                if order.side == OrderSide.SELL and price <= order.stop_price:
                    triggered.append(order)
                elif order.side == OrderSide.BUY and price >= order.stop_price:
                    triggered.append(order)

        for order in triggered:
            self._open_orders.remove(order)
            self._execute_at_price(order, price)

    def cancel_order(self, order_id: str) -> bool:
        for order in self._open_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self._open_orders.remove(order)
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol, Position(symbol=symbol, quantity=0, avg_price=0))

    def get_all_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.quantity != 0]

    def get_account_value(self) -> float:
        equity = self.cash
        for pos in self._positions.values():
            equity += pos.unrealized_pnl
        return equity

    def get_buying_power(self) -> float:
        return self.cash

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        if since:
            return [f for f in self._fills if f.timestamp >= since]
        return list(self._fills)

    def get_open_orders(self) -> list[Order]:
        return list(self._open_orders)

    def get_trade_log(self) -> list[dict]:
        """Get a formatted trade log for analysis."""
        return [
            {
                "order_id": f.order_id,
                "timestamp": f.timestamp.isoformat(),
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.fill_price,
                "commission": f.commission,
            }
            for f in self._fills
        ]
