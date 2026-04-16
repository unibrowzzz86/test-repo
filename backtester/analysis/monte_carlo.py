"""Monte Carlo simulation for strategy robustness testing.

Shuffles trade returns to generate thousands of alternate equity curves,
giving confidence intervals for key metrics. Answers the question:
"How much of this performance was skill vs. luck?"
"""

import pandas as pd
import numpy as np


class MonteCarloSimulation:
    """Monte Carlo simulation by reshuffling trade outcomes."""

    def __init__(self, num_simulations: int = 1000, seed: int = 42):
        self.num_simulations = num_simulations
        self.seed = seed

    def run(
        self,
        trades: pd.DataFrame,
        initial_capital: float = 50000.0,
    ) -> dict:
        """Run Monte Carlo simulation.

        Reshuffles the order of trade P&Ls to generate alternative
        equity paths and compute confidence intervals.

        Args:
            trades: DataFrame with 'pnl' column from backtest.
            initial_capital: Starting capital.

        Returns:
            Dict with simulation statistics.
        """
        if trades.empty:
            return {"error": "No trades to simulate"}

        rng = np.random.RandomState(self.seed)
        pnls = trades["pnl"].values
        n_trades = len(pnls)

        # Storage for simulation results
        final_equities = np.zeros(self.num_simulations)
        max_drawdowns = np.zeros(self.num_simulations)
        max_drawdown_durations = np.zeros(self.num_simulations)
        sharpe_ratios = np.zeros(self.num_simulations)

        for sim in range(self.num_simulations):
            # Shuffle trade order
            shuffled = rng.permutation(pnls)

            # Build equity curve
            equity = np.zeros(n_trades + 1)
            equity[0] = initial_capital
            for i, pnl in enumerate(shuffled):
                equity[i + 1] = equity[i] + pnl

            final_equities[sim] = equity[-1]

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / np.where(peak > 0, peak, 1) * 100
            max_drawdowns[sim] = abs(dd.min())

            # Simple Sharpe approximation from trade returns
            trade_returns = shuffled / initial_capital
            if trade_returns.std() > 0:
                sharpe_ratios[sim] = (
                    trade_returns.mean() / trade_returns.std() * np.sqrt(252 / max(1, n_trades))
                )

        # Compute percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]

        return {
            "num_simulations": self.num_simulations,
            "num_trades": n_trades,
            "original_final_equity": round(initial_capital + pnls.sum(), 2),
            # Final equity distribution
            "final_equity_mean": round(final_equities.mean(), 2),
            "final_equity_median": round(np.median(final_equities), 2),
            "final_equity_std": round(final_equities.std(), 2),
            "final_equity_percentiles": {
                p: round(np.percentile(final_equities, p), 2) for p in percentiles
            },
            "probability_of_profit": round(
                (final_equities > initial_capital).sum() / self.num_simulations * 100, 1
            ),
            # Drawdown distribution
            "max_drawdown_mean_pct": round(max_drawdowns.mean(), 2),
            "max_drawdown_median_pct": round(np.median(max_drawdowns), 2),
            "max_drawdown_95th_pct": round(np.percentile(max_drawdowns, 95), 2),
            "max_drawdown_percentiles": {
                p: round(np.percentile(max_drawdowns, p), 2) for p in percentiles
            },
            # Sharpe distribution
            "sharpe_mean": round(sharpe_ratios.mean(), 3),
            "sharpe_median": round(np.median(sharpe_ratios), 3),
            "sharpe_5th": round(np.percentile(sharpe_ratios, 5), 3),
            # Risk of ruin (equity drops below 50% of initial)
            "risk_of_ruin_pct": round(
                (final_equities < initial_capital * 0.5).sum() / self.num_simulations * 100, 2
            ),
            # Raw distributions for plotting
            "final_equity_distribution": final_equities,
            "max_drawdown_distribution": max_drawdowns,
        }
