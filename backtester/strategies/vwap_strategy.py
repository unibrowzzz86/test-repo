"""VWAP-based intraday strategy for ES futures.

Uses Volume Weighted Average Price as a dynamic support/resistance level.
Combines VWAP with RSI for entry timing and uses ATR for risk management.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class VWAPStrategy(Strategy):
    """VWAP reversion + trend strategy.

    Mode 1 (Reversion): When price deviates significantly from VWAP,
    fade the move expecting reversion.
    Mode 2 (Trend): When price crosses VWAP with momentum, follow the trend.

    Uses RSI to distinguish mean-reversion vs trending conditions.
    """

    def __init__(
        self,
        vwap_dev_threshold: float = 1.5,
        rsi_period: int = 14,
        rsi_trend_threshold: float = 60.0,
        rsi_reversion_threshold: float = 30.0,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        use_reversion_mode: bool = True,
        use_trend_mode: bool = True,
    ):
        super().__init__("VWAP_Strategy")
        self._params = {
            "vwap_dev_threshold": vwap_dev_threshold,
            "rsi_period": rsi_period,
            "rsi_trend_threshold": rsi_trend_threshold,
            "rsi_reversion_threshold": rsi_reversion_threshold,
            "atr_period": atr_period,
            "atr_stop_multiplier": atr_stop_multiplier,
            "use_reversion_mode": use_reversion_mode,
            "use_trend_mode": use_trend_mode,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self._params
        close = data["close"]

        # Rolling VWAP (approximation for daily bars)
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap_20 = (typical_price * data["volume"]).rolling(20).sum() / data["volume"].rolling(20).sum()

        # VWAP standard deviation bands
        vwap_diff = close - vwap_20
        vwap_std = vwap_diff.rolling(20).std()

        rsi_vals = self.rsi(close, p["rsi_period"])
        atr_val = self.atr(data, p["atr_period"])

        signals = pd.Series(0, index=data.index)
        position = 0
        stop_price = 0.0

        warmup = 30

        for i in range(warmup, len(data)):
            price = close.iloc[i]
            atr_now = atr_val.iloc[i]
            rsi_now = rsi_vals.iloc[i]

            if pd.isna(atr_now) or pd.isna(vwap_20.iloc[i]) or pd.isna(vwap_std.iloc[i]):
                signals.iloc[i] = position
                continue

            if vwap_std.iloc[i] == 0:
                signals.iloc[i] = position
                continue

            vwap_z = vwap_diff.iloc[i] / vwap_std.iloc[i]

            # Check stop
            if position == 1 and price < stop_price:
                position = 0
            elif position == -1 and price > stop_price:
                position = 0

            if position == 0:
                # Reversion mode: fade extreme deviations from VWAP
                if p["use_reversion_mode"]:
                    if vwap_z < -p["vwap_dev_threshold"] and rsi_now < p["rsi_reversion_threshold"]:
                        position = 1
                        stop_price = price - p["atr_stop_multiplier"] * atr_now
                    elif vwap_z > p["vwap_dev_threshold"] and rsi_now > (100 - p["rsi_reversion_threshold"]):
                        position = -1
                        stop_price = price + p["atr_stop_multiplier"] * atr_now

                # Trend mode: follow VWAP crossover with momentum
                if position == 0 and p["use_trend_mode"]:
                    prev_z = vwap_diff.iloc[i - 1] / vwap_std.iloc[i - 1] if vwap_std.iloc[i - 1] != 0 else 0
                    if prev_z <= 0 and vwap_z > 0 and rsi_now > p["rsi_trend_threshold"]:
                        position = 1
                        stop_price = price - p["atr_stop_multiplier"] * atr_now
                    elif prev_z >= 0 and vwap_z < 0 and rsi_now < (100 - p["rsi_trend_threshold"]):
                        position = -1
                        stop_price = price + p["atr_stop_multiplier"] * atr_now

            elif position == 1:
                # Trail stop and exit if VWAP z-score reverses
                stop_price = max(stop_price, price - p["atr_stop_multiplier"] * atr_now)
                if vwap_z > p["vwap_dev_threshold"]:
                    position = 0
            elif position == -1:
                stop_price = min(stop_price, price + p["atr_stop_multiplier"] * atr_now)
                if vwap_z < -p["vwap_dev_threshold"]:
                    position = 0

            signals.iloc[i] = position

        return signals
