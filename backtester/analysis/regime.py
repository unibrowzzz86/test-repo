"""Regime analysis for strategy performance.

Breaks down strategy performance by market regime:
trending, mean-reverting, high-vol, low-vol. Helps understand
when the strategy works and when it fails.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class RegimeAnalysis:
    """Analyze strategy performance across different market regimes."""

    def __init__(
        self,
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        vol_high_threshold: float = 1.3,
        vol_low_threshold: float = 0.7,
    ):
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold

    def classify_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar into a market regime.

        Regimes:
        - 'trending_up': Strong uptrend with moderate vol
        - 'trending_down': Strong downtrend with moderate vol
        - 'high_vol': High volatility (any direction)
        - 'low_vol_range': Low volatility, range-bound
        - 'normal': None of the above
        """
        close = data["close"]
        returns = close.pct_change()

        # Rolling volatility
        rolling_vol = returns.rolling(self.vol_lookback).std()
        avg_vol = rolling_vol.expanding().mean()
        vol_ratio = rolling_vol / avg_vol

        # Trend strength (slope of MA)
        ma = close.rolling(self.trend_lookback).mean()
        ma_slope = ma.pct_change(5)  # 5-day slope of MA

        regimes = pd.Series("normal", index=data.index)

        for i in range(max(self.trend_lookback, self.vol_lookback), len(data)):
            vr = vol_ratio.iloc[i]
            slope = ma_slope.iloc[i]

            if pd.isna(vr) or pd.isna(slope):
                continue

            if vr > self.vol_high_threshold:
                regimes.iloc[i] = "high_vol"
            elif vr < self.vol_low_threshold:
                regimes.iloc[i] = "low_vol_range"
            elif slope > 0.003:
                regimes.iloc[i] = "trending_up"
            elif slope < -0.003:
                regimes.iloc[i] = "trending_down"

        return regimes

    def analyze(self, trades: pd.DataFrame, data: pd.DataFrame) -> dict:
        """Break down trade performance by market regime.

        Args:
            trades: Trade log from backtester.
            data: OHLCV data.

        Returns:
            Dict with per-regime performance metrics.
        """
        if trades.empty:
            return {}

        regimes = self.classify_regimes(data)
        results = {}

        # Assign regime to each trade (based on entry date)
        trade_regimes = []
        for _, trade in trades.iterrows():
            entry = trade["entry_date"]
            if entry in regimes.index:
                trade_regimes.append(regimes.loc[entry])
            else:
                # Find nearest date
                idx = regimes.index.get_indexer([entry], method="nearest")[0]
                trade_regimes.append(regimes.iloc[idx])

        trades_with_regime = trades.copy()
        trades_with_regime["regime"] = trade_regimes

        for regime in ["trending_up", "trending_down", "high_vol", "low_vol_range", "normal"]:
            regime_trades = trades_with_regime[trades_with_regime["regime"] == regime]

            if regime_trades.empty:
                results[regime] = {
                    "num_trades": 0,
                    "total_pnl": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "profit_factor": 0,
                }
                continue

            wins = regime_trades[regime_trades["pnl"] > 0]
            losses = regime_trades[regime_trades["pnl"] < 0]
            gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0

            results[regime] = {
                "num_trades": len(regime_trades),
                "total_pnl": round(regime_trades["pnl"].sum(), 2),
                "win_rate": round(len(wins) / len(regime_trades) * 100, 1),
                "avg_pnl": round(regime_trades["pnl"].mean(), 2),
                "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf"),
                "pct_of_trades": round(len(regime_trades) / len(trades) * 100, 1),
            }

        return results
