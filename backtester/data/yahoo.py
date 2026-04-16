"""Yahoo Finance data provider for real ES futures data.

Pulls real OHLCV data using yfinance. Supports daily and intraday
bars for ES, NQ, and other futures/equities.

Usage:
    from backtester.data.yahoo import YahooDataProvider

    # Daily bars
    data = YahooDataProvider.get_es_daily("2025-01-01", "2025-04-15")

    # Intraday 30-min bars (max 60 days history from Yahoo)
    data = YahooDataProvider.get_es_intraday(interval="30m", period="60d")

Requires: pip install yfinance
"""

import pandas as pd
import numpy as np


class YahooDataProvider:
    """Pull real market data from Yahoo Finance."""

    # Common futures tickers
    ES = "ES=F"      # E-mini S&P 500
    NQ = "NQ=F"      # E-mini Nasdaq 100
    YM = "YM=F"      # E-mini Dow
    RTY = "RTY=F"    # E-mini Russell 2000
    CL = "CL=F"      # Crude Oil
    GC = "GC=F"      # Gold
    ZB = "ZB=F"      # 30-Year Treasury Bond

    @staticmethod
    def get_es_daily(
        start_date: str = "2025-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get daily ES futures bars.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string. None = today.

        Returns:
            DataFrame with columns [open, high, low, close, volume].
        """
        return YahooDataProvider.get_data(
            ticker="ES=F",
            start=start_date,
            end=end_date,
            interval="1d",
        )

    @staticmethod
    def get_es_intraday(
        interval: str = "30m",
        period: str = "60d",
    ) -> pd.DataFrame:
        """Get intraday ES futures bars.

        Note: Yahoo limits intraday history to ~60 days for
        30m/1h bars, 7 days for 1m/5m bars.

        Args:
            interval: Bar size - "1m", "5m", "15m", "30m", "1h".
            period: Lookback period - "7d", "30d", "60d".

        Returns:
            DataFrame with columns [open, high, low, close, volume].
        """
        return YahooDataProvider.get_data(
            ticker="ES=F",
            interval=interval,
            period=period,
        )

    @staticmethod
    def get_data(
        ticker: str = "ES=F",
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        period: str | None = None,
    ) -> pd.DataFrame:
        """Generic data pull from Yahoo Finance.

        Args:
            ticker: Yahoo Finance ticker symbol.
            start: Start date (for daily/weekly).
            end: End date.
            interval: Bar interval.
            period: Lookback period (alternative to start/end for intraday).

        Returns:
            Normalized OHLCV DataFrame.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for real data. Install with: pip install yfinance"
            )

        tkr = yf.Ticker(ticker)

        if period:
            df = tkr.history(period=period, interval=interval)
        else:
            df = tkr.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Keep only OHLCV
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in Yahoo data")

        df = df[required].copy()
        df.index.name = "date"

        # Drop any rows with NaN
        df = df.dropna()

        # Convert volume to int
        df["volume"] = df["volume"].astype(int)

        return df

    @staticmethod
    def get_multiple(
        tickers: list[str],
        start: str = "2025-01-01",
        end: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Pull data for multiple tickers."""
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = YahooDataProvider.get_data(
                    ticker=ticker, start=start, end=end, interval=interval,
                )
            except Exception as e:
                print(f"Warning: Failed to get {ticker}: {e}")
        return result
