"""Momentum / Breakout strategy for ES futures.

Trades breakouts from consolidation ranges with volume confirmation.
Uses opening range breakout (ORB) concepts adapted to daily bars,
plus momentum filters (MACD, ROC) to confirm breakout direction.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class MomentumBreakout(Strategy):
    """Breakout strategy with momentum and volume confirmation.

    Entry: Price breaks above/below N-bar high/low with above-average volume
           and MACD confirms direction.
    Exit: Trailing ATR stop or time-based exit.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        volume_threshold: float = 1.3,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,
        max_hold_bars: int = 15,
    ):
        super().__init__("Momentum_Breakout")
        self._params = {
            "lookback_period": lookback_period,
            "volume_threshold": volume_threshold,
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
            "atr_period": atr_period,
            "atr_stop_multiplier": atr_stop_multiplier,
            "max_hold_bars": max_hold_bars,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self._params
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Indicators
        highest_high = high.rolling(window=p["lookback_period"]).max()
        lowest_low = low.rolling(window=p["lookback_period"]).min()
        avg_volume = volume.rolling(window=p["lookback_period"]).mean()
        macd_line, signal_line, macd_hist = self.macd(
            close, p["macd_fast"], p["macd_slow"], p["macd_signal"]
        )
        atr_val = self.atr(data, p["atr_period"])

        signals = pd.Series(0, index=data.index)
        position = 0
        stop_price = 0.0
        bars_in_trade = 0

        warmup = max(p["lookback_period"], p["macd_slow"] + p["macd_signal"]) + 5

        for i in range(warmup, len(data)):
            price = close.iloc[i]
            atr_now = atr_val.iloc[i]

            if pd.isna(atr_now) or atr_now == 0:
                signals.iloc[i] = position
                continue

            # Check exits
            if position != 0:
                bars_in_trade += 1
                # Time exit
                if bars_in_trade >= p["max_hold_bars"]:
                    position = 0
                    bars_in_trade = 0
                # Stop loss
                elif position == 1 and price < stop_price:
                    position = 0
                    bars_in_trade = 0
                elif position == -1 and price > stop_price:
                    position = 0
                    bars_in_trade = 0
                # Trail stop
                elif position == 1:
                    stop_price = max(stop_price, price - p["atr_stop_multiplier"] * atr_now)
                elif position == -1:
                    stop_price = min(stop_price, price + p["atr_stop_multiplier"] * atr_now)

            # New entries
            if position == 0:
                vol_ok = volume.iloc[i] > avg_volume.iloc[i] * p["volume_threshold"]

                # Breakout above range high
                if (
                    close.iloc[i] > highest_high.iloc[i - 1]
                    and vol_ok
                    and macd_hist.iloc[i] > 0
                ):
                    position = 1
                    stop_price = price - p["atr_stop_multiplier"] * atr_now
                    bars_in_trade = 0
                # Breakdown below range low
                elif (
                    close.iloc[i] < lowest_low.iloc[i - 1]
                    and vol_ok
                    and macd_hist.iloc[i] < 0
                ):
                    position = -1
                    stop_price = price + p["atr_stop_multiplier"] * atr_now
                    bars_in_trade = 0

            signals.iloc[i] = position

        return signals
