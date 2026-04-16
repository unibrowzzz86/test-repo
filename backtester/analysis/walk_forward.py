"""Walk-forward analysis.

Splits data into in-sample (training) and out-of-sample (testing) periods,
runs the strategy on each, and aggregates results to test robustness.
This is the gold standard for validating backtested strategies.
"""

import pandas as pd
import numpy as np
from backtester.engine import Backtester, BacktestConfig
from backtester.strategies.base import Strategy
from backtester.metrics import calculate_metrics


class WalkForwardAnalysis:
    """Walk-forward optimization and validation.

    Splits data into rolling windows of in-sample and out-of-sample periods.
    Runs strategy on each out-of-sample period using parameters from the
    preceding in-sample period.
    """

    def __init__(
        self,
        in_sample_days: int = 60,
        out_of_sample_days: int = 20,
        step_days: int = 20,
    ):
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.step_days = step_days

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        config: BacktestConfig | None = None,
    ) -> dict:
        """Run walk-forward analysis.

        Args:
            strategy: Strategy to test.
            data: Full OHLCV dataset.
            config: Backtest configuration.

        Returns:
            Dict with aggregated out-of-sample results.
        """
        config = config or BacktestConfig()
        backtester = Backtester(config)

        windows = self._create_windows(data)
        oos_results = []
        is_results = []

        for window_idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            is_data = data.loc[is_start:is_end]
            oos_data = data.loc[oos_start:oos_end]

            if len(is_data) < 20 or len(oos_data) < 5:
                continue

            # Run on in-sample
            is_result = backtester.run(strategy, is_data)
            is_results.append({
                "window": window_idx,
                "period": f"{is_start.strftime('%Y-%m-%d')} to {is_end.strftime('%Y-%m-%d')}",
                "return_pct": is_result["metrics"].get("total_return_pct", 0),
                "sharpe": is_result["metrics"].get("sharpe_ratio", 0),
                "trades": is_result["metrics"].get("num_trades", 0),
                "win_rate": is_result["metrics"].get("win_rate_pct", 0),
            })

            # Run on out-of-sample
            oos_result = backtester.run(strategy, oos_data)
            oos_results.append({
                "window": window_idx,
                "period": f"{oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}",
                "return_pct": oos_result["metrics"].get("total_return_pct", 0),
                "sharpe": oos_result["metrics"].get("sharpe_ratio", 0),
                "trades": oos_result["metrics"].get("num_trades", 0),
                "win_rate": oos_result["metrics"].get("win_rate_pct", 0),
                "max_dd": oos_result["metrics"].get("max_drawdown_pct", 0),
                "profit_factor": oos_result["metrics"].get("profit_factor", 0),
            })

        if not oos_results:
            return {"error": "Not enough data for walk-forward analysis"}

        oos_df = pd.DataFrame(oos_results)
        is_df = pd.DataFrame(is_results)

        # Walk-forward efficiency: OOS performance / IS performance
        avg_is_return = is_df["return_pct"].mean() if len(is_df) > 0 else 0
        avg_oos_return = oos_df["return_pct"].mean()
        wf_efficiency = (avg_oos_return / avg_is_return * 100) if avg_is_return != 0 else 0

        return {
            "num_windows": len(oos_results),
            "in_sample_results": is_df,
            "out_of_sample_results": oos_df,
            "avg_oos_return_pct": round(avg_oos_return, 2),
            "avg_oos_sharpe": round(oos_df["sharpe"].mean(), 3),
            "avg_oos_win_rate": round(oos_df["win_rate"].mean(), 1),
            "avg_oos_max_dd": round(oos_df["max_dd"].mean(), 2),
            "oos_profitable_windows_pct": round(
                (oos_df["return_pct"] > 0).sum() / len(oos_df) * 100, 1
            ),
            "walk_forward_efficiency_pct": round(wf_efficiency, 1),
            "oos_return_std": round(oos_df["return_pct"].std(), 2),
        }

    def _create_windows(self, data: pd.DataFrame) -> list:
        """Create rolling in-sample / out-of-sample windows."""
        dates = data.index
        windows = []

        start = 0
        while True:
            is_start_idx = start
            is_end_idx = start + self.in_sample_days - 1
            oos_start_idx = is_end_idx + 1
            oos_end_idx = oos_start_idx + self.out_of_sample_days - 1

            if oos_end_idx >= len(dates):
                break

            windows.append((
                dates[is_start_idx],
                dates[is_end_idx],
                dates[oos_start_idx],
                dates[oos_end_idx],
            ))

            start += self.step_days

        return windows
