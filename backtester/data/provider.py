"""Data provider for loading and generating OHLCV data.

Supports loading from CSV files and generating realistic
synthetic ES futures data for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataProvider:
    """Load or generate OHLCV market data."""

    @staticmethod
    def from_csv(filepath: str, date_column: str = "date") -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        Expected columns: date/datetime, open, high, low, close, volume.
        """
        df = pd.read_csv(filepath, parse_dates=[date_column])
        df = df.set_index(date_column)
        df.index.name = "date"

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df = df.sort_index()
        return df[required]

    @staticmethod
    def generate_es_data(
        start_date: str = "2025-01-01",
        end_date: str = "2025-04-15",
        initial_price: float = 5950.0,
        volatility: float = 0.012,
        trend: float = 0.0003,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate realistic synthetic ES futures daily data.

        Creates data with realistic properties:
        - Trending periods and mean-reverting periods
        - Volatility clustering (GARCH-like)
        - Volume patterns (higher on volatile days)
        - Realistic OHLC relationships
        """
        rng = np.random.RandomState(seed)

        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)

        # Generate returns with regime switching and vol clustering
        returns = np.zeros(n)
        vol = np.zeros(n)
        vol[0] = volatility

        # Regime: 0 = trending, 1 = mean-reverting
        regime = np.zeros(n, dtype=int)

        for i in range(1, n):
            # GARCH(1,1) volatility
            vol[i] = np.sqrt(
                0.00001  # omega
                + 0.1 * returns[i - 1] ** 2  # alpha
                + 0.85 * vol[i - 1] ** 2  # beta
            )
            vol[i] = np.clip(vol[i], volatility * 0.3, volatility * 3.0)

            # Regime switching
            if rng.random() < 0.05:
                regime[i] = 1 - regime[i - 1]
            else:
                regime[i] = regime[i - 1]

            # Generate return
            if regime[i] == 0:  # Trending
                returns[i] = trend + vol[i] * rng.randn()
            else:  # Mean-reverting
                mr_pull = -0.3 * returns[i - 1]
                returns[i] = mr_pull + vol[i] * 0.7 * rng.randn()

        # Build price series
        prices = np.zeros(n)
        prices[0] = initial_price
        for i in range(1, n):
            prices[i] = prices[i - 1] * (1 + returns[i])

        # Generate OHLC from close prices
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        volumes = np.zeros(n, dtype=int)

        for i in range(n):
            intraday_vol = vol[i] * prices[i]

            # Open near previous close (with overnight gap)
            if i > 0:
                gap = rng.randn() * intraday_vol * 0.3
                opens[i] = prices[i - 1] + gap
            else:
                opens[i] = prices[i] + rng.randn() * intraday_vol * 0.1

            # High and low relative to open and close
            range_factor = abs(rng.randn()) * 0.5 + 0.5
            bar_range = intraday_vol * range_factor

            high_ext = abs(rng.randn()) * bar_range * 0.5
            low_ext = abs(rng.randn()) * bar_range * 0.5

            highs[i] = max(opens[i], prices[i]) + high_ext
            lows[i] = min(opens[i], prices[i]) - low_ext

            # Volume: higher on volatile days, typical ES volume ~1.5M
            base_volume = 1_500_000
            vol_factor = vol[i] / volatility
            volumes[i] = int(base_volume * vol_factor * (0.7 + 0.6 * rng.random()))

        # Round to ES tick size (0.25)
        def round_tick(arr):
            return np.round(arr / 0.25) * 0.25

        df = pd.DataFrame({
            "open": round_tick(opens),
            "high": round_tick(highs),
            "low": round_tick(lows),
            "close": round_tick(prices),
            "volume": volumes,
        }, index=dates)

        return df

    @staticmethod
    def generate_intraday_es_data(
        start_date: str = "2025-01-01",
        end_date: str = "2025-04-15",
        initial_price: float = 5950.0,
        bar_minutes: int = 30,
        volatility: float = 0.002,
        trend: float = 0.00005,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic ES futures intraday data.

        Creates 30-minute (or custom) bars during ES regular trading hours
        (9:30 AM - 4:00 PM ET). Produces enough bars for strategies to
        generate hundreds of trades, matching the OP's ~347 trades in 3.5 months.
        """
        rng = np.random.RandomState(seed)

        # Generate business days
        business_days = pd.bdate_range(start=start_date, end=end_date)

        # ES regular trading hours: 9:30 AM - 4:00 PM ET
        all_timestamps = []
        for day in business_days:
            market_open = day.replace(hour=9, minute=30)
            market_close = day.replace(hour=16, minute=0)
            timestamps = pd.date_range(
                start=market_open, end=market_close,
                freq=f"{bar_minutes}min",
            )
            # Exclude the close timestamp (we want bars that start before close)
            timestamps = timestamps[timestamps < market_close]
            all_timestamps.extend(timestamps)

        dates = pd.DatetimeIndex(all_timestamps)
        n = len(dates)

        if n == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Generate returns with GARCH-like vol clustering
        returns = np.zeros(n)
        vol = np.zeros(n)
        vol[0] = volatility
        regime = np.zeros(n, dtype=int)

        for i in range(1, n):
            # Check if new day (overnight)
            is_new_day = dates[i].date() != dates[i - 1].date()

            # Vol clustering
            vol[i] = np.sqrt(
                0.0000001
                + 0.08 * returns[i - 1] ** 2
                + 0.88 * vol[i - 1] ** 2
            )
            vol[i] = np.clip(vol[i], volatility * 0.2, volatility * 4.0)

            # Regime switching (more frequent at intraday level)
            if rng.random() < 0.03:
                regime[i] = 1 - regime[i - 1]
            else:
                regime[i] = regime[i - 1]

            # Higher vol at open and close
            bar_hour = dates[i].hour
            bar_minute = dates[i].minute
            time_vol_mult = 1.0
            if bar_hour == 9 and bar_minute == 30:
                time_vol_mult = 1.8  # Opening volatility
            elif bar_hour == 15 and bar_minute >= 30:
                time_vol_mult = 1.4  # Closing volatility
            elif bar_hour < 11:
                time_vol_mult = 1.2  # Morning activity

            effective_vol = vol[i] * time_vol_mult

            # Overnight gap
            if is_new_day:
                gap = rng.randn() * volatility * 2.0
                returns[i] = gap + effective_vol * rng.randn()
            elif regime[i] == 0:  # Trending
                returns[i] = trend + effective_vol * rng.randn()
            else:  # Mean-reverting
                mr_pull = -0.4 * returns[i - 1]
                returns[i] = mr_pull + effective_vol * 0.6 * rng.randn()

        # Build price series
        prices = np.zeros(n)
        prices[0] = initial_price
        for i in range(1, n):
            prices[i] = prices[i - 1] * (1 + returns[i])

        # Build OHLC
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        volumes = np.zeros(n, dtype=int)

        for i in range(n):
            bar_vol = vol[i] * prices[i]

            if i > 0 and dates[i].date() == dates[i - 1].date():
                opens[i] = prices[i - 1] + rng.randn() * bar_vol * 0.1
            elif i > 0:
                opens[i] = prices[i - 1] + rng.randn() * bar_vol * 0.5
            else:
                opens[i] = prices[i]

            ext_h = abs(rng.randn()) * bar_vol * 0.6
            ext_l = abs(rng.randn()) * bar_vol * 0.6

            highs[i] = max(opens[i], prices[i]) + ext_h
            lows[i] = min(opens[i], prices[i]) - ext_l

            # Volume: U-shaped intraday pattern
            bar_hour = dates[i].hour
            if bar_hour <= 10 or bar_hour >= 15:
                base_vol = 120_000
            elif bar_hour <= 12:
                base_vol = 60_000
            else:
                base_vol = 80_000

            volumes[i] = int(base_vol * (0.5 + rng.random()))

        def round_tick(arr):
            return np.round(arr / 0.25) * 0.25

        return pd.DataFrame({
            "open": round_tick(opens),
            "high": round_tick(highs),
            "low": round_tick(lows),
            "close": round_tick(prices),
            "volume": volumes,
        }, index=dates)

    @staticmethod
    def generate_multi_regime_data(
        start_date: str = "2024-01-01",
        end_date: str = "2025-04-15",
        initial_price: float = 4800.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate longer-term data with distinct market regimes.

        Creates data spanning bull runs, corrections, high-vol events,
        and choppy sideways markets - good for stress-testing strategies.
        """
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range(start=start_date, end=end_date)
        n = len(dates)

        prices = np.zeros(n)
        prices[0] = initial_price
        vol = 0.01

        # Define regimes by date ranges
        for i in range(1, n):
            day_of_year = dates[i].timetuple().tm_yday
            month = dates[i].month

            # Regime logic based on time of year
            if month in [1, 2]:  # New year rally
                drift = 0.0008
                vol = 0.009
            elif month == 3:  # Volatile spring
                drift = 0.0004
                vol = 0.015
            elif month in [4, 5]:  # Grind higher
                drift = 0.0005
                vol = 0.010
            elif month in [6, 7]:  # Summer chop
                drift = 0.0001
                vol = 0.008
            elif month == 8:  # August volatility
                drift = -0.0002
                vol = 0.018
            elif month in [9, 10]:  # Fall correction risk
                drift = 0.0002
                vol = 0.014
            elif month in [11, 12]:  # Year-end rally
                drift = 0.0006
                vol = 0.011

            # Add some randomness to regime
            vol *= 0.8 + 0.4 * rng.random()
            ret = drift + vol * rng.randn()
            prices[i] = prices[i - 1] * (1 + ret)

        # Build OHLCV
        opens = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)
        volumes = np.zeros(n, dtype=int)

        for i in range(n):
            intraday_range = prices[i] * vol
            if i > 0:
                opens[i] = prices[i - 1] + rng.randn() * intraday_range * 0.2
            else:
                opens[i] = prices[i]

            ext_h = abs(rng.randn()) * intraday_range * 0.5
            ext_l = abs(rng.randn()) * intraday_range * 0.5
            highs[i] = max(opens[i], prices[i]) + ext_h
            lows[i] = min(opens[i], prices[i]) - ext_l
            volumes[i] = int(1_500_000 * (0.7 + 0.6 * rng.random()))

        def round_tick(arr):
            return np.round(arr / 0.25) * 0.25

        return pd.DataFrame({
            "open": round_tick(opens),
            "high": round_tick(highs),
            "low": round_tick(lows),
            "close": round_tick(prices),
            "volume": volumes,
        }, index=dates)
