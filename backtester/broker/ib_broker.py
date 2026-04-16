"""Interactive Brokers live execution broker.

Connects to IB TWS or IB Gateway via the ib_insync library.
Handles order submission, position tracking, and account monitoring
for live ES futures trading.

Setup:
    1. Install IB TWS or IB Gateway
    2. Enable API connections in TWS: File > Global Config > API > Settings
       - Enable ActiveX and Socket Clients
       - Socket port: 7497 (paper) or 7496 (live)
       - Trusted IPs: 127.0.0.1
    3. pip install ib_insync

Usage:
    from backtester.broker.ib_broker import IBBroker

    broker = IBBroker(host="127.0.0.1", port=7497)  # 7497=paper, 7496=live
    broker.connect()

    # Submit order
    order = Order(symbol="ES", side=OrderSide.BUY, quantity=1)
    broker.submit_order(order)
"""

import logging
from datetime import datetime
from backtester.broker.base import (
    Broker, Order, Fill, Position, OrderSide, OrderType, OrderStatus,
)

logger = logging.getLogger(__name__)


class IBBroker(Broker):
    """Interactive Brokers broker implementation.

    Wraps ib_insync for clean order management and position tracking.
    """

    # ES futures contract specs
    ES_EXCHANGE = "CME"
    ES_CURRENCY = "USD"
    ES_MULTIPLIER = "50"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 = paper trading, 7496 = live
        client_id: int = 1,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self._fills: list[Fill] = []
        self._order_map: dict[int, str] = {}  # IB order ID -> our order ID
        self._next_order_id = 1

    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True
        except ImportError:
            raise ImportError(
                "ib_insync is required for IB trading. Install with: pip install ib_insync"
            )
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        return self.ib is not None and self.ib.isConnected()

    def _make_es_contract(self):
        """Create an ES futures contract."""
        from ib_insync import Future
        return Future(
            symbol="ES",
            exchange=self.ES_EXCHANGE,
            currency=self.ES_CURRENCY,
        )

    def _to_ib_order(self, order: Order):
        """Convert our Order to an IB order."""
        from ib_insync import MarketOrder, LimitOrder, StopOrder, Order as IBOrder

        action = "BUY" if order.side == OrderSide.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            return MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            return LimitOrder(action, order.quantity, order.limit_price)
        elif order.order_type == OrderType.STOP:
            return StopOrder(action, order.quantity, order.stop_price)
        else:
            # Stop-limit
            ib_order = IBOrder()
            ib_order.action = action
            ib_order.totalQuantity = order.quantity
            ib_order.orderType = "STP LMT"
            ib_order.auxPrice = order.stop_price
            ib_order.lmtPrice = order.limit_price
            return ib_order

    def submit_order(self, order: Order) -> str:
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        contract = self._make_es_contract()

        # Qualify the contract (resolve to specific expiry)
        self.ib.qualifyContracts(contract)

        ib_order = self._to_ib_order(order)
        trade = self.ib.placeOrder(contract, ib_order)

        order_id = f"IB_{self._next_order_id}"
        self._next_order_id += 1
        self._order_map[trade.order.orderId] = order_id

        logger.info(
            f"Order submitted: {order_id} {order.side.value} {order.quantity} ES "
            f"@ {order.order_type.value}"
        )

        # Register fill callback
        trade.filledEvent += lambda t: self._on_fill(t, order_id)

        return order_id

    def _on_fill(self, trade, order_id: str):
        """Handle fill notification from IB."""
        for fill_event in trade.fills:
            f = Fill(
                order_id=order_id,
                symbol="ES",
                side=OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL,
                quantity=int(fill_event.execution.shares),
                fill_price=fill_event.execution.price,
                commission=fill_event.commissionReport.commission if fill_event.commissionReport else 0,
                timestamp=datetime.now(),
            )
            self._fills.append(f)
            logger.info(
                f"Fill: {order_id} {f.side.value} {f.quantity} @ {f.fill_price} "
                f"(commission: ${f.commission:.2f})"
            )

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected():
            return False
        for ib_id, our_id in self._order_map.items():
            if our_id == order_id:
                for trade in self.ib.openTrades():
                    if trade.order.orderId == ib_id:
                        self.ib.cancelOrder(trade.order)
                        logger.info(f"Order cancelled: {order_id}")
                        return True
        return False

    def get_position(self, symbol: str) -> Position | None:
        if not self.is_connected():
            return None
        for pos in self.ib.positions():
            if pos.contract.symbol == symbol:
                return Position(
                    symbol=symbol,
                    quantity=int(pos.position),
                    avg_price=pos.avgCost / float(self.ES_MULTIPLIER),
                    unrealized_pnl=0,  # Calculated separately
                )
        return Position(symbol=symbol, quantity=0, avg_price=0)

    def get_all_positions(self) -> list[Position]:
        if not self.is_connected():
            return []
        positions = []
        for pos in self.ib.positions():
            positions.append(Position(
                symbol=pos.contract.symbol,
                quantity=int(pos.position),
                avg_price=pos.avgCost,
            ))
        return positions

    def get_account_value(self) -> float:
        if not self.is_connected():
            return 0
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        return 0

    def get_buying_power(self) -> float:
        if not self.is_connected():
            return 0
        for av in self.ib.accountValues():
            if av.tag == "BuyingPower" and av.currency == "USD":
                return float(av.value)
        return 0

    def get_fills(self, since: datetime | None = None) -> list[Fill]:
        if since:
            return [f for f in self._fills if f.timestamp >= since]
        return list(self._fills)

    def get_open_orders(self) -> list[Order]:
        if not self.is_connected():
            return []
        orders = []
        for trade in self.ib.openTrades():
            side = OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL
            ib_id = trade.order.orderId
            our_id = self._order_map.get(ib_id, f"IB_{ib_id}")
            orders.append(Order(
                symbol=trade.contract.symbol,
                side=side,
                quantity=int(trade.order.totalQuantity),
                order_id=our_id,
                status=OrderStatus.SUBMITTED,
            ))
        return orders

    def get_real_time_price(self, symbol: str = "ES") -> float | None:
        """Get the latest price for a contract."""
        if not self.is_connected():
            return None
        contract = self._make_es_contract()
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(2)  # Wait for data
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        return float(price) if price == price else None  # NaN check
