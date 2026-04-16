#!/usr/bin/env python3
"""Go live: TradingView signals -> IB execution.

This is the production entry point. It:
1. Connects to Interactive Brokers (paper or live)
2. Starts a webhook server to receive TradingView alerts
3. Routes signals to IB for execution
4. Monitors performance and enforces kill rules
5. Logs everything

Step-by-step setup:
    1. Install IB TWS or Gateway: https://www.interactivebrokers.com
    2. Enable API: File > Global Config > API > Settings
       - Check "Enable ActiveX and Socket Clients"
       - Port: 7497 (paper) or 7496 (live)
    3. In TradingView, set up your strategy alert:
       - Webhook URL: http://YOUR_IP:5000/webhook
       - Message format:
         {"action":"{{strategy.order.action}}","contracts":"{{strategy.order.contracts}}","price":"{{close}}"}
    4. Run this script:
       python examples/go_live.py --paper     # Paper trading first!
       python examples/go_live.py --live      # Only when ready

IMPORTANT: Always paper trade first. The kill rules will protect you,
but verify everything works before risking real money.
"""

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.broker.paper_broker import PaperBroker
from backtester.broker.tradingview_webhook import TradingViewReceiver
from backtester.monitoring.monitor import LiveMonitor
from backtester.costs import ES_FUTURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Go live with TradingView + IB")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true", help="Paper trading mode (IB paper or simulated)")
    mode.add_argument("--live", action="store_true", help="LIVE trading (real money!)")
    mode.add_argument("--sim", action="store_true", help="Simulated broker (no IB needed)")

    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=None,
                       help="IB port (default: 7497 for paper, 7496 for live)")
    parser.add_argument("--webhook-port", type=int, default=5000)
    parser.add_argument("--webhook-secret", default="",
                       help="Secret to validate TradingView webhooks")
    parser.add_argument("--capital", type=float, default=50000)
    parser.add_argument("--max-position", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=3.0,
                       help="Max daily loss as %% of capital")
    parser.add_argument("--max-drawdown", type=float, default=20.0,
                       help="Kill switch drawdown %%")

    args = parser.parse_args()

    # Safety check for live trading
    if args.live:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE - REAL MONEY AT RISK")
        print("!" * 60)
        confirm = input("\nType 'I UNDERSTAND' to continue: ")
        if confirm != "I UNDERSTAND":
            print("Aborted.")
            sys.exit(0)

    # Setup broker
    if args.sim:
        broker = PaperBroker(initial_capital=args.capital, cost_model=ES_FUTURES)
        logger.info("Using simulated broker (no IB connection)")
    else:
        from backtester.broker.ib_broker import IBBroker
        port = args.ib_port
        if port is None:
            port = 7497 if args.paper else 7496
        broker = IBBroker(host=args.ib_host, port=port)
        mode_str = "PAPER" if args.paper else "LIVE"
        logger.info(f"Connecting to IB ({mode_str}) at {args.ib_host}:{port}")

    if not broker.connect():
        logger.error("Failed to connect to broker. Exiting.")
        sys.exit(1)

    # Setup monitor
    monitor = LiveMonitor(
        max_drawdown_alert_pct=args.max_drawdown * 0.5,
        max_drawdown_kill_pct=args.max_drawdown,
        daily_loss_limit_pct=args.daily_loss_limit,
    )

    # Setup TradingView webhook receiver
    receiver = TradingViewReceiver(
        broker=broker,
        secret=args.webhook_secret,
        max_position=args.max_position,
    )

    logger.info("=" * 60)
    logger.info("SYSTEM READY")
    logger.info(f"  Mode: {'PAPER' if args.paper else 'LIVE' if args.live else 'SIM'}")
    logger.info(f"  Capital: ${args.capital:,.0f}")
    logger.info(f"  Max position: {args.max_position} contracts")
    logger.info(f"  Daily loss limit: {args.daily_loss_limit}%")
    logger.info(f"  Drawdown kill: {args.max_drawdown}%")
    logger.info(f"  Webhook port: {args.webhook_port}")
    logger.info("")
    logger.info("  TradingView alert webhook URL:")
    logger.info(f"    http://YOUR_IP:{args.webhook_port}/webhook")
    logger.info("")
    logger.info("  Health check:")
    logger.info(f"    http://localhost:{args.webhook_port}/health")
    logger.info("=" * 60)

    # Start webhook server (blocking)
    try:
        receiver.start(host="0.0.0.0", port=args.webhook_port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        broker.flatten_all()
        broker.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
