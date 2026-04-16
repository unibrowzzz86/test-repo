"""Base strategy interface.

All strategies must inherit from this class and implement
generate_signals(). Strategies receive OHLCV data and return
a signal series: 1 = long, -1 = short, 0 = flat.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Strategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._params: dict = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
                  indexed by datetime.

        Returns:
            Series of signals: 1 (long), -1 (short), 0 (flat).
        """
        pass

    def get_params(self) -> dict:
        return self._params.copy()

    def set_params(self, **params):
        for k, v in params.items():
            if k in self._params:
                self._params[k] = v
        return self

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        return upper, mid, lower

    @staticmethod
    def vwap(data: pd.DataFrame) -> pd.Series:
        """Calculate intraday-style VWAP (cumulative reset is handled externally)."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        cumulative_tp_vol = (typical_price * data["volume"]).cumsum()
        cumulative_vol = data["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
