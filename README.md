# ES Futures Backtesting System

A complete algorithmic trading backtesting framework for ES futures, inspired by the r/algotrading community. Includes multiple trading strategies, realistic cost modeling, comprehensive performance metrics, robustness analysis, and an interactive Streamlit dashboard.

## Quick Start

```bash
pip install -r requirements.txt

# Run the CLI backtest
python examples/run_backtest.py

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

## Features

### Trading Strategies
- **SMA/EMA Crossover** - Trend-following with ATR trailing stops and trend filter
- **Mean Reversion** - RSI + Bollinger Bands for fading overextended moves
- **Momentum Breakout** - Range breakouts with volume confirmation and MACD
- **VWAP Strategy** - VWAP reversion and trend modes with RSI timing
- **Ensemble** - Weighted voting system combining all strategies

### Backtesting Engine
- Bar-by-bar simulation with realistic execution
- Configurable transaction costs (commissions, slippage, spread, market impact)
- Signal delay modeling (gap between signal and fill)
- Position sizing with kill rules
- Support for daily and intraday (30-min) bars

### Performance Metrics
- Total return, net profit, CAGR
- Win rate, profit factor, payoff ratio
- Sharpe, Sortino, Calmar ratios
- Max drawdown (%, $, duration, recovery time)
- Monthly/yearly breakdown
- Win/loss streaks
- Loss clustering analysis (worst 3/5-trade sequences)

### Robustness Analysis
- **Walk-Forward Analysis** - In-sample/out-of-sample validation
- **Monte Carlo Simulation** - 1000 reshuffled equity curves for confidence intervals
- **Regime Analysis** - Performance breakdown by market regime (trending, range-bound, high-vol)

### Risk Management
- Daily loss limits (auto-stops trading for the day)
- Drawdown-based position size reduction
- Drawdown kill switch (halts strategy entirely)
- Exposure monitoring (gross, net, sector concentration)

### Live Monitoring
- Real-time performance tracking vs backtest benchmarks
- Win rate degradation alerts
- Consecutive loss alerts
- Drawdown alerts with kill switch
- Go-live readiness checklist

### Interactive Dashboard
- Dark-themed Streamlit UI matching the r/algotrading aesthetic
- Equity curve and drawdown charts
- Trade distribution histogram
- Monthly P&L breakdown bars
- Walk-forward, Monte Carlo, and regime analysis tabs
- Full trade log and configuration export

## Project Structure

```
├── backtester/
│   ├── engine.py              # Core backtesting engine
│   ├── metrics.py             # Performance metrics
│   ├── costs.py               # Transaction cost modeling
│   ├── position_sizing.py     # Position sizing & kill rules
│   ├── strategies/
│   │   ├── base.py            # Strategy base class (indicators included)
│   │   ├── sma_crossover.py   # Trend-following strategy
│   │   ├── mean_reversion.py  # Mean reversion strategy
│   │   ├── momentum_breakout.py # Breakout strategy
│   │   ├── vwap_strategy.py   # VWAP-based strategy
│   │   └── ensemble.py        # Multi-strategy ensemble
│   ├── data/
│   │   └── provider.py        # Data loading & synthetic generation
│   ├── analysis/
│   │   ├── walk_forward.py    # Walk-forward analysis
│   │   ├── monte_carlo.py     # Monte Carlo simulation
│   │   └── regime.py          # Market regime analysis
│   └── monitoring/
│       └── monitor.py         # Live monitoring & alerts
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── examples/
│   └── run_backtest.py        # Example CLI usage
├── tests/
│   └── test_backtester.py     # Test suite (29 tests)
└── requirements.txt
```

## Using Your Own Data

```python
from backtester.data.provider import DataProvider

# Load from CSV (expects columns: date, open, high, low, close, volume)
data = DataProvider.from_csv("your_es_data.csv")
```

## Writing Custom Strategies

```python
from backtester.strategies.base import Strategy

class MyStrategy(Strategy):
    def generate_signals(self, data):
        # Use built-in indicators: self.sma(), self.ema(), self.rsi(),
        # self.atr(), self.bollinger_bands(), self.macd(), self.vwap()
        signals = pd.Series(0, index=data.index)
        # Your logic here: 1 = long, -1 = short, 0 = flat
        return signals
```

## Go-Live Checklist (from r/algotrading)

Before going live, verify:

1. **Edge is real** - Out-of-sample results, walk-forward efficiency > 50%, performance across regimes
2. **Realistic costs** - Commissions, slippage, spread, market impact, signal delay
3. **Risk profile** - Max drawdown, worst month, loss clusters
4. **Position sizing** - Capital per trade, daily loss limit, drawdown kill switch
5. **Implementation** - Clean data, correct order routing, no look-ahead bias
6. **Monitoring** - Win rate tracking, drawdown alerts, consecutive loss limits
