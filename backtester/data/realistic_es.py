"""High-fidelity ES futures data generator.

Generates synthetic ES data that closely replicates real market
microstructure:
- Realistic price levels (~5900-6100 range for early 2025)
- Intraday U-shaped volume pattern
- Volatility clustering (GARCH)
- Opening gaps and overnight sessions
- Trend/chop regime switching
- Realistic tick-level noise

This is NOT real data — use YahooDataProvider or PolygonDataProvider
for actual market data. This exists for strategy development and
testing when you don't have a data subscription yet.
"""

import pandas as pd
import numpy as np


def generate_realistic_es_5min(
    start_date: str = "2025-01-02",
    end_date: str = "2025-04-15",
    initial_price: float = 5950.0,
    seed: int = 12345,
) -> pd.DataFrame:
    """Generate 5-minute ES bars that mimic real market behavior.

    5-min bars produce ~5000+ bars in a 3.5 month period,
    giving strategies plenty of data to generate 300+ trades
    like the OP's system.

    Returns:
        OHLCV DataFrame indexed by datetime.
    """
    rng = np.random.RandomState(seed)

    # ES regular trading hours: 9:30 AM - 4:00 PM ET = 78 five-minute bars/day
    business_days = pd.bdate_range(start=start_date, end=end_date)

    all_timestamps = []
    for day in business_days:
        market_open = day.replace(hour=9, minute=30)
        market_close = day.replace(hour=16, minute=0)
        timestamps = pd.date_range(start=market_open, end=market_close, freq="5min")
        timestamps = timestamps[timestamps < market_close]
        all_timestamps.extend(timestamps)

    dates = pd.DatetimeIndex(all_timestamps)
    n = len(dates)

    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Price generation with realistic microstructure
    prices = np.zeros(n)
    prices[0] = initial_price
    vol = np.zeros(n)
    vol[0] = 0.0008  # ~0.08% per 5-min bar baseline

    # Track regime per day
    day_regime = {}  # date -> regime
    current_regime = 0  # 0=trending, 1=choppy

    for i in range(1, n):
        current_date = dates[i].date()
        prev_date = dates[i - 1].date()
        is_new_day = current_date != prev_date

        # Regime switching at daily level
        if is_new_day:
            if rng.random() < 0.35:  # 35% chance of regime switch
                current_regime = 1 - current_regime
            day_regime[current_date] = current_regime

        # GARCH volatility
        vol[i] = np.sqrt(
            1e-8 + 0.12 * (prices[i-1] * 0.0008 * rng.randn()) ** 2 / prices[i-1]**2
            + 0.85 * vol[i-1] ** 2
        )
        vol[i] = np.clip(vol[i], 0.0002, 0.003)

        # Time-of-day volatility multiplier (U-shape)
        bar_hour = dates[i].hour
        bar_minute = dates[i].minute
        minutes_from_open = (bar_hour - 9) * 60 + (bar_minute - 30)
        minutes_to_close = (16 * 60) - (bar_hour * 60 + bar_minute)

        if minutes_from_open < 30:
            time_mult = 2.0  # First 30 min: high vol
        elif minutes_from_open < 60:
            time_mult = 1.5
        elif minutes_to_close < 30:
            time_mult = 1.8  # Last 30 min: high vol
        elif minutes_to_close < 60:
            time_mult = 1.3
        elif 11 * 60 + 30 <= bar_hour * 60 + bar_minute <= 13 * 60:
            time_mult = 0.6  # Lunch: low vol
        else:
            time_mult = 1.0

        effective_vol = vol[i] * time_mult

        # Overnight gap
        if is_new_day:
            # Gap based on "overnight sentiment"
            gap_size = rng.randn() * 0.004  # ~0.4% average gap
            # Slight upward bias (bull market early 2025)
            gap_size += 0.0005
            prices[i] = prices[i - 1] * (1 + gap_size)
        else:
            # Intraday return
            regime = day_regime.get(current_date, 0)

            if regime == 0:  # Trending day
                # Slight momentum continuation
                if i >= 2:
                    momentum = 0.15 * (prices[i-1] - prices[i-2]) / prices[i-2]
                else:
                    momentum = 0
                drift = 0.00003 + momentum
                ret = drift + effective_vol * rng.randn()
            else:  # Choppy/mean-reverting day
                if i >= 2:
                    mean_rev = -0.25 * (prices[i-1] - prices[i-2]) / prices[i-2]
                else:
                    mean_rev = 0
                ret = mean_rev + effective_vol * 0.8 * rng.randn()

            prices[i] = prices[i - 1] * (1 + ret)

    # Generate OHLC from close prices
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.zeros(n, dtype=int)

    for i in range(n):
        bar_vol_pts = vol[i] * prices[i]

        # Open
        if i > 0 and dates[i].date() == dates[i-1].date():
            opens[i] = prices[i-1] + rng.randn() * bar_vol_pts * 0.05
        elif i > 0:
            opens[i] = prices[i]  # Gap already in price
        else:
            opens[i] = prices[i]

        # High/Low with realistic bar structure
        bar_range = bar_vol_pts * (0.5 + abs(rng.randn()) * 0.5)
        body = abs(prices[i] - opens[i])

        # Wick proportions
        upper_wick = abs(rng.randn()) * bar_range * 0.3
        lower_wick = abs(rng.randn()) * bar_range * 0.3

        highs[i] = max(opens[i], prices[i]) + upper_wick
        lows[i] = min(opens[i], prices[i]) - lower_wick

        # Volume: U-shaped intraday + random variation
        bar_hour = dates[i].hour
        bar_minute = dates[i].minute
        minutes_from_open = (bar_hour - 9) * 60 + (bar_minute - 30)
        minutes_to_close = (16 * 60) - (bar_hour * 60 + bar_minute)

        if minutes_from_open < 15:
            base_vol = 45000
        elif minutes_from_open < 30:
            base_vol = 35000
        elif minutes_to_close < 15:
            base_vol = 40000
        elif minutes_to_close < 30:
            base_vol = 30000
        elif 11 * 60 <= bar_hour * 60 + bar_minute <= 13 * 60:
            base_vol = 12000
        else:
            base_vol = 20000

        # Add randomness and volatility correlation
        vol_mult = vol[i] / 0.0008  # Higher vol = higher volume
        volumes[i] = int(base_vol * vol_mult * (0.6 + 0.8 * rng.random()))

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


def generate_realistic_es_1min(
    start_date: str = "2025-01-02",
    end_date: str = "2025-04-15",
    initial_price: float = 5950.0,
    seed: int = 12345,
) -> pd.DataFrame:
    """Generate 1-minute ES bars for scalping strategies.

    1-min bars produce ~25,000+ bars in 3.5 months.
    This is what you'd use for the kind of scalping strategy
    that generates 347 trades with ~$300 avg win.
    """
    rng = np.random.RandomState(seed)

    business_days = pd.bdate_range(start=start_date, end=end_date)

    all_timestamps = []
    for day in business_days:
        market_open = day.replace(hour=9, minute=30)
        market_close = day.replace(hour=16, minute=0)
        timestamps = pd.date_range(start=market_open, end=market_close, freq="1min")
        timestamps = timestamps[timestamps < market_close]
        all_timestamps.extend(timestamps)

    dates = pd.DatetimeIndex(all_timestamps)
    n = len(dates)

    prices = np.zeros(n)
    prices[0] = initial_price
    vol = np.zeros(n)
    vol[0] = 0.00025

    for i in range(1, n):
        is_new_day = dates[i].date() != dates[i-1].date()

        # GARCH vol
        vol[i] = np.sqrt(1e-9 + 0.1 * vol[i-1]**2 * rng.randn()**2 + 0.87 * vol[i-1]**2)
        vol[i] = np.clip(vol[i], 0.00005, 0.001)

        # Time-of-day multiplier
        bar_hour = dates[i].hour
        bar_minute = dates[i].minute
        minutes_from_open = (bar_hour - 9) * 60 + (bar_minute - 30)
        minutes_to_close = 390 - minutes_from_open

        if minutes_from_open < 15:
            time_mult = 2.5
        elif minutes_from_open < 30:
            time_mult = 1.8
        elif minutes_to_close < 15:
            time_mult = 2.0
        elif minutes_to_close < 30:
            time_mult = 1.5
        elif 120 <= minutes_from_open <= 210:
            time_mult = 0.5
        else:
            time_mult = 1.0

        if is_new_day:
            gap = rng.randn() * 0.004 + 0.0003
            prices[i] = prices[i-1] * (1 + gap)
        else:
            ret = 0.000005 + vol[i] * time_mult * rng.randn()
            # Slight mean reversion at 1-min level
            if i >= 5:
                recent_ret = (prices[i-1] - prices[i-5]) / prices[i-5]
                ret -= 0.05 * recent_ret
            prices[i] = prices[i-1] * (1 + ret)

    # Build OHLCV
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    volumes = np.zeros(n, dtype=int)

    for i in range(n):
        bv = vol[i] * prices[i]
        opens[i] = prices[i-1] if i > 0 and dates[i].date() == dates[i-1].date() else prices[i]
        uw = abs(rng.randn()) * bv * 0.4
        lw = abs(rng.randn()) * bv * 0.4
        highs[i] = max(opens[i], prices[i]) + uw
        lows[i] = min(opens[i], prices[i]) - lw

        bar_hour = dates[i].hour
        minutes_from_open = (bar_hour - 9) * 60 + (dates[i].minute - 30)
        if minutes_from_open < 15:
            bvol = 12000
        elif minutes_from_open > 375:
            bvol = 10000
        elif 120 <= minutes_from_open <= 210:
            bvol = 2500
        else:
            bvol = 5000
        volumes[i] = int(bvol * (0.5 + rng.random()))

    def round_tick(arr):
        return np.round(arr / 0.25) * 0.25

    return pd.DataFrame({
        "open": round_tick(opens),
        "high": round_tick(highs),
        "low": round_tick(lows),
        "close": round_tick(prices),
        "volume": volumes,
    }, index=dates)
