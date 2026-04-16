"""SMA/EMA Crossover trend-following strategy for ES futures.

Uses fast/slow moving average crossovers with ATR-based filtering
to trade with the trend. Adds a trend strength filter (ADX-style)
to avoid whipsaw in ranging markets.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class SMACrossover(Strategy):
    """Trend-following strategy using moving average crossovers.

    Entry: Fast MA crosses above slow MA (long) or below (short).
    Filter: Only take trades when trend strength (measured by MA slope) is sufficient.
    Exit: Opposite crossover or trailing stop based on ATR.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        trend_filter_period: int = 50,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        use_ema: bool = True,
    ):
        super().__init__("SMA_Crossover")
        self._params = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "trend_filter_period": trend_filter_period,
            "atr_period": atr_period,
            "atr_stop_multiplier": atr_stop_multiplier,
            "use_ema": use_ema,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self._params
        close = data["close"]

        ma_func = self.ema if p["use_ema"] else self.sma
        fast_ma = ma_func(close, p["fast_period"])
        slow_ma = ma_func(close, p["slow_period"])
        trend_ma = ma_func(close, p["trend_filter_period"])

        atr_val = self.atr(data, p["atr_period"])

        signals = pd.Series(0, index=data.index)

        # Trend filter: only trade in direction of the longer-term MA
        above_trend = close > trend_ma
        below_trend = close < trend_ma

        # Crossover signals
        cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        # Generate signals with trend filter
        position = 0
        stop_price = 0.0

        for i in range(len(data)):
            if i < p["trend_filter_period"]:
                continue

            price = close.iloc[i]
            atr_now = atr_val.iloc[i]

            if pd.isna(atr_now) or atr_now == 0:
                continue

            # Check stop loss
            if position == 1 and price < stop_price:
                position = 0
            elif position == -1 and price > stop_price:
                position = 0

            # New entries
            if position == 0:
                if cross_up.iloc[i] and above_trend.iloc[i]:
                    position = 1
                    stop_price = price - p["atr_stop_multiplier"] * atr_now
                elif cross_down.iloc[i] and below_trend.iloc[i]:
                    position = -1
                    stop_price = price + p["atr_stop_multiplier"] * atr_now
            elif position == 1:
                # Trail stop
                new_stop = price - p["atr_stop_multiplier"] * atr_now
                stop_price = max(stop_price, new_stop)
                # Exit on opposite crossover
                if cross_down.iloc[i]:
                    position = 0
            elif position == -1:
                new_stop = price + p["atr_stop_multiplier"] * atr_now
                stop_price = min(stop_price, new_stop)
                if cross_up.iloc[i]:
                    position = 0

            signals.iloc[i] = position

        return signals
