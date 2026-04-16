# ES Futures Trading System

A production-ready algorithmic trading system for ES futures. Backtest strategies, run walk-forward analysis, and go live with Interactive Brokers execution and TradingView signal integration.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Backtest on real data (pulls from Yahoo Finance)
python examples/run_real_data.py --source yahoo --strategy scalper

# 2. Backtest on TradingView CSV export
python examples/run_real_data.py --source csv --file your_data.csv

# 3. Paper trade with TradingView webhooks
python examples/go_live.py --paper

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

## Architecture

```
backtester/
├── engine.py                  # Core backtesting engine
├── metrics.py                 # 30+ performance metrics
├── costs.py                   # Transaction cost modeling
├── position_sizing.py         # Position sizing & kill rules
├── live_runner.py             # Live trading execution loop
├── strategies/
│   ├── base.py                # Strategy base class (built-in indicators)
│   ├── es_scalper.py          # ES scalping strategy (5-min bars)
│   ├── sma_crossover.py       # Trend-following
│   ├── mean_reversion.py      # RSI + Bollinger Bands
│   ├── momentum_breakout.py   # Range breakout with volume
│   ├── vwap_strategy.py       # VWAP reversion/trend
│   └── ensemble.py            # Multi-strategy voting
├── data/
│   ├── provider.py            # CSV loader + synthetic data
│   ├── yahoo.py               # Yahoo Finance (free, real data)
│   ├── polygon_provider.py    # Polygon.io (paid, tick-level)
│   └── realistic_es.py        # High-fidelity ES simulator
├── broker/
│   ├── base.py                # Broker interface
│   ├── ib_broker.py           # Interactive Brokers execution
│   ├── paper_broker.py        # Paper trading (same interface)
│   └── tradingview_webhook.py # TradingView alert receiver
├── analysis/
│   ├── walk_forward.py        # Walk-forward validation
│   ├── monte_carlo.py         # Monte Carlo simulation
│   └── regime.py              # Market regime analysis
└── monitoring/
    └── monitor.py             # Live monitoring & alerts
```

## Data Sources

| Source | Cost | Data | Best For |
|--------|------|------|----------|
| **Yahoo Finance** | Free | Daily + 60d intraday | Quick backtests |
| **Polygon.io** | Free tier / Paid | Tick to daily | Production backtests |
| **TradingView CSV** | Free with account | Any timeframe | Your own exports |
| **Interactive Brokers** | Account required | Real-time | Live trading |

```python
# Yahoo Finance
from backtester.data.yahoo import YahooDataProvider
data = YahooDataProvider.get_es_intraday(interval="30m", period="60d")

# Polygon.io
from backtester.data.polygon_provider import PolygonDataProvider
provider = PolygonDataProvider(api_key="YOUR_KEY")
data = provider.get_es_bars("2025-01-01", "2025-04-15", timespan="minute", multiplier=5)

# CSV (TradingView export)
from backtester.data.provider import DataProvider
data = DataProvider.from_csv("es_data.csv")
```

## Going Live

### Option 1: TradingView Signals → IB Execution

Use your existing TradingView strategies. Alerts fire webhooks that route to IB.

```bash
# Step 1: Paper trade first (ALWAYS)
python examples/go_live.py --paper --webhook-port 5000

# Step 2: In TradingView, create alert with webhook:
#   URL: http://YOUR_IP:5000/webhook
#   Message: {"action":"{{strategy.order.action}}","contracts":"1","price":"{{close}}"}

# Step 3: When confident, go live
python examples/go_live.py --live --webhook-port 5000
```

### Option 2: Autonomous Strategy Execution

Run the built-in strategies directly against IB.

```bash
# Paper trade the ensemble strategy
python -m backtester.live_runner --mode paper --strategy ensemble

# Paper trade the scalper
python -m backtester.live_runner --mode paper --strategy scalper

# Live with IB (port 7496)
python -m backtester.live_runner --mode ib --port 7496 --strategy scalper
```

### Risk Management (Always Active)

- **Daily loss limit**: Stops trading for the day (default: 3%)
- **Drawdown reduction**: Cuts position size at 10% DD
- **Kill switch**: Halts all trading at 20% DD
- **EOD flatten**: Closes all positions at market close
- **Consecutive loss alert**: Warns at 8 losses in a row

## Writing Custom Strategies

```python
from backtester.strategies.base import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__("MyStrategy")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        # Built-in indicators
        fast = self.ema(close, 8)
        slow = self.ema(close, 21)
        rsi_val = self.rsi(close, 14)
        atr_val = self.atr(data, 14)
        upper, mid, lower = self.bollinger_bands(close, 20)
        macd_line, signal, histogram = self.macd(close)

        signals = pd.Series(0, index=data.index)
        # Your logic: 1 = long, -1 = short, 0 = flat
        return signals
```

## Robustness Checklist (from r/algotrading)

Before going live, verify all six:

1. **Edge is real** — Walk-forward efficiency > 50%, profitable across regimes, 100+ trades
2. **Realistic costs** — Commissions, slippage, spread, signal delay all modeled
3. **Risk profile** — Survive worst month, worst 5-trade sequence, max DD
4. **Position sizing** — Capital per trade, daily loss limit, drawdown kill switch
5. **Implementation** — Clean data, correct order routing, no look-ahead bias
6. **Monitoring** — Win rate alerts, drawdown alerts, consecutive loss limits
