"""ES Futures scalping strategy.

Designed for 5-minute bars, targeting ~6 point average wins on ES.
Combines multiple signals to find high-probability short-term entries:

1. VWAP deviation for mean-reversion scalps
2. EMA momentum for trend scalps
3. RSI divergence for reversal timing
4. ATR-based dynamic stops and targets

This is the type of strategy that could produce the OP's results:
~347 trades in 3.5 months = ~4-5 trades per day, with a 55% win rate
and 1.5:1 payoff ratio.
"""

import pandas as pd
import numpy as np
from backtester.strategies.base import Strategy


class ESScalper(Strategy):
    """High-frequency scalper for ES futures on 5-min bars.

    Trades both trend continuation and mean-reversion setups
    depending on market conditions. Uses ATR-based stops
    with fixed R:R targets.
    """

    def __init__(
        self,
        fast_ema: int = 8,
        slow_ema: int = 21,
        rsi_period: int = 6,
        rsi_entry_low: float = 30.0,
        rsi_entry_high: float = 70.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.2,
        atr_target_mult: float = 1.8,
        vwap_period: int = 78,  # ~1 full trading day of 5-min bars
        vwap_dev_entry: float = 1.0,
        trend_ema: int = 50,
        max_hold_bars: int = 30,  # 2.5 hours max hold
        cooldown_bars: int = 3,
    ):
        super().__init__("ES_Scalper")
        self._params = {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "rsi_period": rsi_period,
            "rsi_entry_low": rsi_entry_low,
            "rsi_entry_high": rsi_entry_high,
            "atr_period": atr_period,
            "atr_stop_mult": atr_stop_mult,
            "atr_target_mult": atr_target_mult,
            "vwap_period": vwap_period,
            "vwap_dev_entry": vwap_dev_entry,
            "trend_ema": trend_ema,
            "max_hold_bars": max_hold_bars,
            "cooldown_bars": cooldown_bars,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        p = self._params
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Indicators
        fast = self.ema(close, p["fast_ema"])
        slow = self.ema(close, p["slow_ema"])
        trend = self.ema(close, p["trend_ema"])
        rsi_val = self.rsi(close, p["rsi_period"])
        atr_val = self.atr(data, p["atr_period"])

        # Rolling VWAP
        typical = (high + low + close) / 3
        vwap_tp_vol = (typical * volume).rolling(p["vwap_period"]).sum()
        vwap_vol = volume.rolling(p["vwap_period"]).sum()
        vwap_line = vwap_tp_vol / vwap_vol.replace(0, np.nan)
        vwap_std = (close - vwap_line).rolling(p["vwap_period"]).std()

        # MACD for momentum confirmation
        macd_line, macd_signal, macd_hist = self.macd(close, 12, 26, 9)

        signals = pd.Series(0, index=data.index)
        position = 0
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        bars_in_trade = 0
        bars_since_exit = 999

        warmup = max(p["trend_ema"], p["vwap_period"]) + 10

        for i in range(warmup, len(data)):
            price = close.iloc[i]
            atr_now = atr_val.iloc[i]
            rsi_now = rsi_val.iloc[i]

            if pd.isna(atr_now) or atr_now == 0 or pd.isna(rsi_now):
                signals.iloc[i] = position
                continue

            if pd.isna(vwap_line.iloc[i]) or pd.isna(vwap_std.iloc[i]) or vwap_std.iloc[i] == 0:
                signals.iloc[i] = position
                continue

            vwap_z = (price - vwap_line.iloc[i]) / vwap_std.iloc[i]

            # Position management
            if position != 0:
                bars_in_trade += 1

                # Check target
                if position == 1 and price >= target_price:
                    position = 0
                    bars_since_exit = 0
                elif position == -1 and price <= target_price:
                    position = 0
                    bars_since_exit = 0
                # Check stop
                elif position == 1 and price <= stop_price:
                    position = 0
                    bars_since_exit = 0
                elif position == -1 and price >= stop_price:
                    position = 0
                    bars_since_exit = 0
                # Time exit
                elif bars_in_trade >= p["max_hold_bars"]:
                    position = 0
                    bars_since_exit = 0
                # Trail stop on winners
                elif position == 1 and price > entry_price + atr_now * 0.5:
                    stop_price = max(stop_price, price - atr_now * 0.8)
                elif position == -1 and price < entry_price - atr_now * 0.5:
                    stop_price = min(stop_price, price + atr_now * 0.8)
            else:
                bars_since_exit += 1

            # Entry logic (only if not in a trade and past cooldown)
            if position == 0 and bars_since_exit >= p["cooldown_bars"]:
                trend_up = fast.iloc[i] > slow.iloc[i] and price > trend.iloc[i]
                trend_down = fast.iloc[i] < slow.iloc[i] and price < trend.iloc[i]
                macd_bull = macd_hist.iloc[i] > 0
                macd_bear = macd_hist.iloc[i] < 0

                # === SETUP 1: Trend continuation pullback ===
                # Price pulled back to slow EMA in an uptrend, RSI oversold
                if (trend_up and rsi_now < p["rsi_entry_low"] + 10
                    and abs(price - slow.iloc[i]) < atr_now * 0.5
                    and macd_bull):
                    position = 1
                    entry_price = price
                    stop_price = price - atr_now * p["atr_stop_mult"]
                    target_price = price + atr_now * p["atr_target_mult"]
                    bars_in_trade = 0

                elif (trend_down and rsi_now > p["rsi_entry_high"] - 10
                      and abs(price - slow.iloc[i]) < atr_now * 0.5
                      and macd_bear):
                    position = -1
                    entry_price = price
                    stop_price = price + atr_now * p["atr_stop_mult"]
                    target_price = price - atr_now * p["atr_target_mult"]
                    bars_in_trade = 0

                # === SETUP 2: VWAP mean reversion ===
                # Price far below VWAP with oversold RSI
                elif (vwap_z < -p["vwap_dev_entry"]
                      and rsi_now < p["rsi_entry_low"]
                      and not trend_down):
                    position = 1
                    entry_price = price
                    stop_price = price - atr_now * p["atr_stop_mult"]
                    target_price = price + atr_now * p["atr_target_mult"]
                    bars_in_trade = 0

                elif (vwap_z > p["vwap_dev_entry"]
                      and rsi_now > p["rsi_entry_high"]
                      and not trend_up):
                    position = -1
                    entry_price = price
                    stop_price = price + atr_now * p["atr_stop_mult"]
                    target_price = price - atr_now * p["atr_target_mult"]
                    bars_in_trade = 0

                # === SETUP 3: EMA crossover with momentum ===
                if position == 0:
                    prev_fast_above = fast.iloc[i-1] > slow.iloc[i-1] if i > 0 else False
                    curr_fast_above = fast.iloc[i] > slow.iloc[i]

                    # Bullish crossover
                    if (not prev_fast_above and curr_fast_above
                        and macd_bull and rsi_now > 40 and rsi_now < 65
                        and price > trend.iloc[i]):
                        position = 1
                        entry_price = price
                        stop_price = price - atr_now * p["atr_stop_mult"]
                        target_price = price + atr_now * p["atr_target_mult"]
                        bars_in_trade = 0

                    # Bearish crossover
                    elif (prev_fast_above and not curr_fast_above
                          and macd_bear and rsi_now < 60 and rsi_now > 35
                          and price < trend.iloc[i]):
                        position = -1
                        entry_price = price
                        stop_price = price + atr_now * p["atr_stop_mult"]
                        target_price = price - atr_now * p["atr_target_mult"]
                        bars_in_trade = 0

            signals.iloc[i] = position

        return signals
