"""TradingView webhook receiver.

Receives alert signals from TradingView strategies via webhook,
translates them into orders, and routes them to your broker.

Setup in TradingView:
    1. Create an alert on your strategy
    2. Set webhook URL to: http://YOUR_SERVER:5000/webhook
    3. Set alert message to JSON:
       {
           "action": "{{strategy.order.action}}",
           "contracts": "{{strategy.order.contracts}}",
           "ticker": "{{ticker}}",
           "price": "{{close}}",
           "time": "{{time}}",
           "strategy": "{{strategy.order.comment}}"
       }

Usage:
    from backtester.broker.tradingview_webhook import TradingViewReceiver

    receiver = TradingViewReceiver(broker=broker, secret="your_secret")
    receiver.start(port=5000)  # Starts Flask server
"""

import json
import logging
import hashlib
import hmac
from datetime import datetime
from threading import Thread

from backtester.broker.base import Broker, Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class TradingViewReceiver:
    """Receives TradingView webhook alerts and converts to orders.

    TradingView sends a POST request to /webhook whenever your
    strategy fires an alert. This receiver validates the signal,
    translates it to an order, and sends it to your broker.
    """

    def __init__(
        self,
        broker: Broker,
        secret: str = "",
        symbol: str = "ES",
        max_position: int = 1,
        allowed_ips: list[str] | None = None,
    ):
        self.broker = broker
        self.secret = secret
        self.symbol = symbol
        self.max_position = max_position
        self.allowed_ips = allowed_ips or []
        self.signals_received: list[dict] = []
        self.app = None

    def start(self, host: str = "0.0.0.0", port: int = 5000):
        """Start the webhook server."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError(
                "Flask is required for TradingView webhooks. "
                "Install with: pip install flask"
            )

        self.app = Flask(__name__)

        @self.app.route("/webhook", methods=["POST"])
        def webhook():
            # IP filtering
            if self.allowed_ips and request.remote_addr not in self.allowed_ips:
                logger.warning(f"Rejected request from {request.remote_addr}")
                return jsonify({"error": "unauthorized"}), 403

            # Parse signal
            try:
                data = request.get_json(force=True)
            except Exception:
                return jsonify({"error": "invalid json"}), 400

            # Validate secret if configured
            if self.secret:
                sig = request.headers.get("X-Webhook-Secret", "")
                if sig != self.secret:
                    logger.warning("Invalid webhook secret")
                    return jsonify({"error": "invalid secret"}), 403

            logger.info(f"TradingView signal received: {data}")
            self.signals_received.append({
                "timestamp": datetime.now().isoformat(),
                "data": data,
            })

            # Process signal
            try:
                result = self._process_signal(data)
                return jsonify(result), 200
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "ok",
                "broker_connected": self.broker.is_connected(),
                "signals_received": len(self.signals_received),
                "position": self._get_position_info(),
            })

        @self.app.route("/signals", methods=["GET"])
        def signals():
            return jsonify(self.signals_received[-50:])  # Last 50

        logger.info(f"Starting TradingView webhook server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False)

    def start_background(self, host: str = "0.0.0.0", port: int = 5000):
        """Start webhook server in a background thread."""
        thread = Thread(target=self.start, args=(host, port), daemon=True)
        thread.start()
        logger.info(f"Webhook server started in background on {host}:{port}")
        return thread

    def _process_signal(self, data: dict) -> dict:
        """Convert a TradingView signal to a broker order."""
        action = data.get("action", "").upper()
        contracts = int(data.get("contracts", 1))
        price = float(data.get("price", 0))

        contracts = min(contracts, self.max_position)

        if not self.broker.is_connected():
            raise ConnectionError("Broker not connected")

        current_pos = self.broker.get_position(self.symbol)
        current_qty = current_pos.quantity if current_pos else 0

        order_id = None
        action_taken = "none"

        if action == "BUY":
            if current_qty < 0:
                # Close short first
                self.broker.flatten(self.symbol)
                action_taken = "closed_short"

            if current_qty <= 0:
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=contracts,
                    order_type=OrderType.MARKET,
                )
                order_id = self.broker.submit_order(order)
                action_taken = "buy_long"

        elif action == "SELL":
            if current_qty > 0:
                # Close long first
                self.broker.flatten(self.symbol)
                action_taken = "closed_long"

            if current_qty >= 0:
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=contracts,
                    order_type=OrderType.MARKET,
                )
                order_id = self.broker.submit_order(order)
                action_taken = "sell_short"

        elif action in ("EXIT", "CLOSE", "FLATTEN"):
            if current_qty != 0:
                order_id = self.broker.flatten(self.symbol)
                action_taken = "flattened"

        else:
            return {"error": f"Unknown action: {action}"}

        return {
            "action_taken": action_taken,
            "order_id": order_id,
            "signal_price": price,
            "previous_position": current_qty,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_position_info(self) -> dict:
        pos = self.broker.get_position(self.symbol)
        if pos:
            return {
                "symbol": self.symbol,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "unrealized_pnl": pos.unrealized_pnl,
            }
        return {"symbol": self.symbol, "quantity": 0}
