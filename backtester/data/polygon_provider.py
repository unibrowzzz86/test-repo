"""Polygon.io data provider for high-quality ES futures data.

Polygon provides tick-level and aggregate bar data for futures.
Free tier gives delayed data; paid tiers give real-time.

Usage:
    from backtester.data.polygon_provider import PolygonDataProvider

    provider = PolygonDataProvider(api_key="YOUR_KEY")
    data = provider.get_es_bars(
        start_date="2025-01-01",
        end_date="2025-04-15",
        timespan="minute",
        multiplier=30,
    )

Requires: pip install requests
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


class PolygonDataProvider:
    """Pull real market data from Polygon.io REST API."""

    BASE_URL = "https://api.polygon.io"

    # ES futures ticker format on Polygon
    ES_FRONT_MONTH = "ESM5"  # June 2025 - update quarterly
    ES_CONTINUOUS = "ES"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_es_bars(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-04-15",
        timespan: str = "minute",
        multiplier: int = 30,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        """Get ES futures aggregate bars from Polygon.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            timespan: "minute", "hour", "day".
            multiplier: Number of timespan units per bar (e.g., 30 for 30-min bars).
            ticker: Specific futures ticker. None = use continuous.

        Returns:
            OHLCV DataFrame.
        """
        ticker = ticker or self.ES_CONTINUOUS
        return self._get_aggregates(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            start_date=start_date,
            end_date=end_date,
        )

    def _get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Pull aggregate bars with pagination."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required. Install with: pip install requests")

        all_results = []
        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{start_date}/{end_date}"
        )
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }

        while url:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                all_results.extend(data["results"])

            # Pagination
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": self.api_key}
            else:
                url = None

            # Rate limit (free tier: 5 req/min)
            time.sleep(0.25)

        if not all_results:
            raise ValueError(f"No data returned for {ticker}")

        df = pd.DataFrame(all_results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date")

        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        })

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["volume"] = df["volume"].astype(int)
        df = df.sort_index()

        return df
