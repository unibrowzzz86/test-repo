"""Ensemble strategy that combines multiple sub-strategies.

Uses a voting or weighted scheme to combine signals from multiple
strategies. This is the approach that likely generated the OP's results -
combining trend, mean reversion, and momentum signals.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class EnsembleStrategy(Strategy):
    """Combines multiple strategies via weighted voting.

    Each sub-strategy gets a weight. The combined signal is the
    weighted sum, thresholded to produce -1, 0, or 1.

    Strategies that disagree cancel out, reducing overtrading
    and improving risk-adjusted returns.
    """

    def __init__(
        self,
        strategies: list[tuple[Strategy, float]] | None = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            strategies: List of (strategy, weight) tuples.
            threshold: Minimum weighted signal to trigger a trade.
        """
        super().__init__("Ensemble")
        self.strategies = strategies or []
        self.threshold = threshold
        self._params = {
            "threshold": threshold,
            "num_strategies": len(self.strategies),
            "strategy_names": [s.name for s, _ in self.strategies],
            "weights": [w for _, w in self.strategies],
        }

    def add_strategy(self, strategy: Strategy, weight: float = 1.0):
        self.strategies.append((strategy, weight))
        self._params["num_strategies"] = len(self.strategies)
        self._params["strategy_names"] = [s.name for s, _ in self.strategies]
        self._params["weights"] = [w for _, w in self.strategies]

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.strategies:
            return pd.Series(0, index=data.index)

        # Collect signals from each sub-strategy
        all_signals = pd.DataFrame(index=data.index)
        total_weight = sum(w for _, w in self.strategies)

        for strategy, weight in self.strategies:
            sig = strategy.generate_signals(data)
            all_signals[strategy.name] = sig * (weight / total_weight)

        # Weighted sum
        combined = all_signals.sum(axis=1)

        # Threshold to discrete signals
        signals = pd.Series(0, index=data.index)
        signals[combined > self.threshold] = 1
        signals[combined < -self.threshold] = -1

        return signals

    def get_sub_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get individual signals from each sub-strategy (for analysis)."""
        sub_signals = pd.DataFrame(index=data.index)
        for strategy, weight in self.strategies:
            sub_signals[strategy.name] = strategy.generate_signals(data)
        return sub_signals
