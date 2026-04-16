"""Core backtesting engine.

Simulates trading with realistic execution, cost modeling,
position sizing, and kill rules. Produces trade logs and
equity curves for analysis.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from backtester.strategies.base import Strategy
from backtester.costs import CostModel, ES_FUTURES
from backtester.position_sizing import PositionSizer
from backtester.metrics import calculate_metrics


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: float = 50000.0
    cost_model: CostModel = field(default_factory=lambda: ES_FUTURES)
    position_sizer: PositionSizer = field(default_factory=PositionSizer)
    tick_value: float = 12.50  # ES futures: $12.50 per tick (0.25 point * $50/point)
    point_value: float = 50.0  # ES futures: $50 per point
    risk_free_rate: float = 0.05
    margin_per_contract: float = 15000.0  # Approximate ES margin


class Backtester:
    """Event-driven backtesting engine.

    Walks through data bar-by-bar, applying strategy signals,
    executing trades with realistic costs, and tracking P&L.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
    ) -> dict:
        """Run a backtest.

        Args:
            strategy: Strategy instance to test.
            data: OHLCV DataFrame indexed by datetime.

        Returns:
            Dict with metrics, trades, equity curve.
        """
        cfg = self.config
        cost = cfg.cost_model
        sizer = cfg.position_sizer

        # Generate signals
        signals = strategy.generate_signals(data)

        # Apply signal delay if configured
        if cost.signal_delay_bars > 0:
            signals = signals.shift(cost.signal_delay_bars).fillna(0).astype(int)

        # Walk through bars
        equity = cfg.initial_capital
        peak_equity = equity
        position = 0
        entry_price = 0.0
        entry_date = None
        daily_pnl = 0.0
        current_date = None

        trades = []
        equity_values = []
        equity_dates = []

        for i in range(len(data)):
            bar = data.iloc[i]
            date = data.index[i]
            price = bar["close"]
            signal = int(signals.iloc[i])

            # Reset daily P&L on new day
            bar_date = date.date() if hasattr(date, "date") else date
            if current_date != bar_date:
                daily_pnl = 0.0
                current_date = bar_date

            # Check kill rules
            dd_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0
            daily_pnl_pct = (daily_pnl / equity * 100) if equity > 0 else 0

            if sizer.should_stop(dd_pct, daily_pnl_pct) and position != 0:
                # Force close
                exit_price = cost.adjusted_exit_price(price, position)
                pnl = (exit_price - entry_price) * position * cfg.point_value
                commission = cost.commission_per_contract
                pnl -= commission

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "quantity": 1,
                    "exit_reason": "kill_rule",
                })
                equity += pnl
                daily_pnl += pnl
                position = 0
                signal = 0  # Override signal

            # Position change needed?
            if signal != position:
                # Close existing position
                if position != 0:
                    exit_price = cost.adjusted_exit_price(price, position)
                    pnl = (exit_price - entry_price) * position * cfg.point_value
                    commission = cost.commission_per_contract
                    pnl -= commission

                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": position,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": round(pnl, 2),
                        "quantity": 1,
                        "exit_reason": "signal",
                    })
                    equity += pnl
                    daily_pnl += pnl
                    position = 0

                # Open new position
                if signal != 0 and not sizer.should_stop(dd_pct, daily_pnl_pct):
                    size = sizer.calculate_size(
                        equity=equity,
                        price=price,
                        risk_per_unit=self.config.point_value * 2,  # Approximate 2-point risk
                        current_drawdown_pct=dd_pct,
                        daily_pnl_pct=daily_pnl_pct,
                    )
                    if size > 0:
                        entry_price = cost.adjusted_entry_price(price, signal)
                        entry_date = date
                        position = signal
                        # Deduct entry commission
                        equity -= cost.commission_per_contract

            # Track equity
            if position != 0:
                # Mark to market
                unrealized = (price - entry_price) * position * cfg.point_value
                equity_values.append(equity + unrealized)
            else:
                equity_values.append(equity)

            equity_dates.append(date)
            peak_equity = max(peak_equity, equity_values[-1])

        # Close any remaining position at last bar
        if position != 0:
            last_price = data["close"].iloc[-1]
            exit_price = cost.adjusted_exit_price(last_price, position)
            pnl = (exit_price - entry_price) * position * cfg.point_value
            pnl -= cost.commission_per_contract

            trades.append({
                "entry_date": entry_date,
                "exit_date": data.index[-1],
                "direction": position,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "quantity": 1,
                "exit_reason": "end_of_data",
            })
            equity_values[-1] = equity + pnl

        # Build results
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

        equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))

        metrics = calculate_metrics(
            trades=trades_df,
            equity_curve=equity_curve,
            initial_capital=cfg.initial_capital,
            risk_free_rate=cfg.risk_free_rate,
        )

        return {
            "metrics": metrics,
            "trades": trades_df,
            "equity_curve": equity_curve,
            "signals": signals,
            "strategy_name": strategy.name,
            "config": {
                "initial_capital": cfg.initial_capital,
                "cost_model": cost.summary(),
                "position_sizing": sizer.summary(),
                "strategy_params": strategy.get_params(),
            },
        }

    def run_multiple(
        self,
        strategies: list[Strategy],
        data: pd.DataFrame,
    ) -> list[dict]:
        """Run backtests for multiple strategies on the same data."""
        return [self.run(strategy, data) for strategy in strategies]

    def compare(
        self,
        results: list[dict],
    ) -> pd.DataFrame:
        """Create a comparison table of multiple backtest results."""
        rows = []
        for r in results:
            m = r["metrics"]
            rows.append({
                "Strategy": r["strategy_name"],
                "Total Return %": m.get("total_return_pct", 0),
                "Net Profit": m.get("net_profit", 0),
                "Trades": m.get("num_trades", 0),
                "Win Rate %": m.get("win_rate_pct", 0),
                "Profit Factor": m.get("profit_factor", 0),
                "Sharpe": m.get("sharpe_ratio", 0),
                "Sortino": m.get("sortino_ratio", 0),
                "Calmar": m.get("calmar_ratio", 0),
                "Max DD %": m.get("max_drawdown_pct", 0),
                "Avg Win": m.get("avg_win", 0),
                "Avg Loss": m.get("avg_loss", 0),
            })
        return pd.DataFrame(rows)
