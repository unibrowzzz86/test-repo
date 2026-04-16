"""Tests for the backtesting engine, metrics, and strategies."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

from backtester.engine import Backtester, BacktestConfig
from backtester.costs import CostModel, ES_FUTURES
from backtester.position_sizing import PositionSizer
from backtester.metrics import calculate_metrics
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import MeanReversion
from backtester.strategies.momentum_breakout import MomentumBreakout
from backtester.strategies.vwap_strategy import VWAPStrategy
from backtester.strategies.ensemble import EnsembleStrategy
from backtester.data.provider import DataProvider
from backtester.analysis.monte_carlo import MonteCarloSimulation
from backtester.analysis.regime import RegimeAnalysis


@pytest.fixture
def sample_data():
    """Generate sample ES data for testing."""
    return DataProvider.generate_es_data(
        start_date="2025-01-01",
        end_date="2025-04-15",
        seed=42,
    )


@pytest.fixture
def config():
    return BacktestConfig(initial_capital=50000.0)


class TestDataProvider:
    def test_generate_es_data_shape(self, sample_data):
        assert len(sample_data) > 50
        assert list(sample_data.columns) == ["open", "high", "low", "close", "volume"]

    def test_generate_es_data_ohlc_valid(self, sample_data):
        assert (sample_data["high"] >= sample_data["low"]).all()
        assert (sample_data["high"] >= sample_data["close"]).all()
        assert (sample_data["low"] <= sample_data["close"]).all()
        assert (sample_data["volume"] > 0).all()

    def test_generate_es_data_tick_alignment(self, sample_data):
        for col in ["open", "high", "low", "close"]:
            remainder = (sample_data[col] * 4).round(0) % 1
            assert (remainder < 0.01).all(), f"{col} not aligned to 0.25 ticks"

    def test_different_seeds_produce_different_data(self):
        d1 = DataProvider.generate_es_data(seed=1)
        d2 = DataProvider.generate_es_data(seed=2)
        assert not d1["close"].equals(d2["close"])

    def test_same_seed_reproduces_data(self):
        d1 = DataProvider.generate_es_data(seed=42)
        d2 = DataProvider.generate_es_data(seed=42)
        assert d1["close"].equals(d2["close"])


class TestCostModel:
    def test_round_trip_cost(self):
        cost = CostModel(commission_per_contract=2.25, slippage_ticks=1.0, tick_size=0.25)
        rt = cost.total_cost_per_trade(5000.0)
        assert rt > 0

    def test_adjusted_entry_price_long(self):
        cost = CostModel(slippage_ticks=1.0, tick_size=0.25, spread_ticks=1.0)
        adjusted = cost.adjusted_entry_price(5000.0, direction=1)
        assert adjusted > 5000.0  # Should be worse (higher) for long

    def test_adjusted_entry_price_short(self):
        cost = CostModel(slippage_ticks=1.0, tick_size=0.25, spread_ticks=1.0)
        adjusted = cost.adjusted_entry_price(5000.0, direction=-1)
        assert adjusted < 5000.0  # Should be worse (lower) for short

    def test_zero_cost_model(self):
        cost = CostModel(
            commission_per_contract=0, slippage_ticks=0,
            spread_ticks=0, market_impact_bps=0,
        )
        assert cost.total_cost_per_trade(5000.0) == 0


class TestPositionSizer:
    def test_normal_sizing(self):
        sizer = PositionSizer(max_position_size=5)
        size = sizer.calculate_size(equity=100000, price=5000, risk_per_unit=100)
        assert 0 < size <= 5

    def test_kill_on_drawdown(self):
        sizer = PositionSizer(drawdown_stop_pct=20.0)
        size = sizer.calculate_size(
            equity=100000, price=5000, risk_per_unit=100,
            current_drawdown_pct=25.0,
        )
        assert size == 0

    def test_reduce_on_drawdown(self):
        sizer = PositionSizer(
            max_position_size=10,
            drawdown_reduce_pct=10.0,
            reduce_size_factor=0.5,
        )
        normal = sizer.calculate_size(equity=100000, price=5000, risk_per_unit=100)
        reduced = sizer.calculate_size(
            equity=100000, price=5000, risk_per_unit=100,
            current_drawdown_pct=15.0,
        )
        assert reduced <= normal

    def test_daily_loss_limit_kills(self):
        sizer = PositionSizer(daily_loss_limit_pct=3.0)
        assert sizer.should_stop(current_drawdown_pct=1.0, daily_pnl_pct=-4.0)


class TestStrategies:
    def test_sma_crossover_generates_signals(self, sample_data):
        strategy = SMACrossover()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_mean_reversion_generates_signals(self, sample_data):
        strategy = MeanReversion()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_momentum_breakout_generates_signals(self, sample_data):
        strategy = MomentumBreakout()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_vwap_generates_signals(self, sample_data):
        strategy = VWAPStrategy()
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_ensemble_generates_signals(self, sample_data):
        ensemble = EnsembleStrategy(threshold=0.3)
        ensemble.add_strategy(SMACrossover(), 1.0)
        ensemble.add_strategy(MeanReversion(), 1.0)
        signals = ensemble.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestBacktester:
    def test_backtest_runs(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)
        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result

    def test_backtest_preserves_capital(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)
        # Equity should never go to zero
        assert result["equity_curve"].min() > 0

    def test_backtest_metrics_consistent(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)
        m = result["metrics"]
        trades = result["trades"]

        if len(trades) > 0:
            assert m["num_trades"] == len(trades)
            assert m["num_wins"] + m["num_losses"] <= m["num_trades"]

    def test_compare_strategies(self, sample_data, config):
        bt = Backtester(config)
        results = bt.run_multiple(
            [SMACrossover(), MeanReversion()],
            sample_data,
        )
        comparison = bt.compare(results)
        assert len(comparison) == 2
        assert "Strategy" in comparison.columns

    def test_cost_model_reduces_returns(self, sample_data):
        # Zero cost
        config_free = BacktestConfig(
            initial_capital=50000.0,
            cost_model=CostModel(
                commission_per_contract=0, slippage_ticks=0,
                spread_ticks=0, market_impact_bps=0,
            ),
        )
        # Normal cost
        config_costly = BacktestConfig(
            initial_capital=50000.0,
            cost_model=ES_FUTURES,
        )

        bt_free = Backtester(config_free)
        bt_costly = Backtester(config_costly)

        r_free = bt_free.run(SMACrossover(), sample_data)
        r_costly = bt_costly.run(SMACrossover(), sample_data)

        # Costs should reduce final equity
        free_pnl = r_free["metrics"].get("net_profit", 0)
        costly_pnl = r_costly["metrics"].get("net_profit", 0)

        if r_free["metrics"]["num_trades"] > 0:
            assert free_pnl >= costly_pnl


class TestMetrics:
    def test_metrics_with_trades(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)
        m = result["metrics"]

        assert "total_return_pct" in m
        assert "sharpe_ratio" in m
        assert "sortino_ratio" in m
        assert "max_drawdown_pct" in m
        assert "win_rate_pct" in m
        assert "profit_factor" in m

    def test_drawdown_is_positive(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)
        assert result["metrics"].get("max_drawdown_pct", 0) >= 0

    def test_win_rate_bounded(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)
        wr = result["metrics"].get("win_rate_pct", 0)
        assert 0 <= wr <= 100


class TestMonteCarlo:
    def test_monte_carlo_runs(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)

        if not result["trades"].empty:
            mc = MonteCarloSimulation(num_simulations=100)
            mc_result = mc.run(result["trades"], config.initial_capital)
            assert "probability_of_profit" in mc_result
            assert 0 <= mc_result["probability_of_profit"] <= 100


class TestRegimeAnalysis:
    def test_regime_classification(self, sample_data):
        ra = RegimeAnalysis()
        regimes = ra.classify_regimes(sample_data)
        assert len(regimes) == len(sample_data)

    def test_regime_analysis_runs(self, sample_data, config):
        bt = Backtester(config)
        result = bt.run(SMACrossover(), sample_data)

        if not result["trades"].empty:
            ra = RegimeAnalysis()
            regime_result = ra.analyze(result["trades"], sample_data)
            assert isinstance(regime_result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
