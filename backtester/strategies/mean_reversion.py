"""Mean reversion strategy for ES futures.

Uses RSI extremes with Bollinger Band confirmation to fade
overextended moves. Works well during range-bound/choppy conditions
that whipsaw trend-following strategies.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class MeanReversion(Strategy):
    """Mean reversion strategy using RSI + Bollinger Bands.

    Entry Long: RSI below oversold threshold AND price touches lower BB.
    Entry Short: RSI above overbought threshold AND price touches upper BB.
    Exit: RSI returns to neutral zone OR price hits ATR-based stop.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        rsi_exit_low: float = 45.0,
        rsi_exit_high: float = 55.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
    ):
        super().__init__("Mean_Reversion")
        self._params = {
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "rsi_exit_low": rsi_exit_low,
            "rsi_exit_high": rsi_exit_high,
            "bb_period": bb_period,
            "bb_std": bb_std,
            "atr_period": atr_period,
            "atr_stop_multiplier": atr_stop_multiplier,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self._params
        close = data["close"]

        rsi_vals = self.rsi(close, p["rsi_period"])
        bb_upper, bb_mid, bb_lower = self.bollinger_bands(close, p["bb_period"], p["bb_std"])
        atr_val = self.atr(data, p["atr_period"])

        signals = pd.Series(0, index=data.index)
        position = 0
        stop_price = 0.0

        warmup = max(p["rsi_period"], p["bb_period"]) + 5

        for i in range(warmup, len(data)):
            price = close.iloc[i]
            rsi_now = rsi_vals.iloc[i]
            atr_now = atr_val.iloc[i]

            if pd.isna(rsi_now) or pd.isna(atr_now) or atr_now == 0:
                signals.iloc[i] = position
                continue

            # Check stop
            if position == 1 and price < stop_price:
                position = 0
            elif position == -1 and price > stop_price:
                position = 0

            if position == 0:
                # Oversold -> go long
                if rsi_now < p["rsi_oversold"] and price <= bb_lower.iloc[i]:
                    position = 1
                    stop_price = price - p["atr_stop_multiplier"] * atr_now
                # Overbought -> go short
                elif rsi_now > p["rsi_overbought"] and price >= bb_upper.iloc[i]:
                    position = -1
                    stop_price = price + p["atr_stop_multiplier"] * atr_now
            elif position == 1:
                # Exit when RSI normalizes or hits mid BB
                if rsi_now > p["rsi_exit_high"] or price >= bb_mid.iloc[i]:
                    position = 0
            elif position == -1:
                if rsi_now < p["rsi_exit_low"] or price <= bb_mid.iloc[i]:
                    position = 0

            signals.iloc[i] = position

        return signals
