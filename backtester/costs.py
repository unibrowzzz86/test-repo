"""Transaction cost modeling with realistic assumptions.

Models commissions, slippage, spread, and market impact
so backtests don't lie about real-world performance.
"""

from dataclasses import dataclass, field


@dataclass
class CostModel:
    """Models all transaction costs for realistic backtesting.

    Attributes:
        commission_per_contract: Fixed commission per contract/share.
        slippage_ticks: Average slippage in price ticks per trade.
        tick_size: Minimum price movement (e.g., 0.25 for ES futures).
        spread_ticks: Bid-ask spread in ticks.
        market_impact_bps: Market impact in basis points (relevant for larger sizes).
        signal_delay_bars: Number of bars delay between signal and fill.
    """

    commission_per_contract: float = 2.25
    slippage_ticks: float = 1.0
    tick_size: float = 0.25
    spread_ticks: float = 1.0
    market_impact_bps: float = 0.0
    signal_delay_bars: int = 0

    def total_cost_per_trade(self, price: float, quantity: int = 1) -> float:
        """Calculate total round-trip cost for a trade."""
        commission = self.commission_per_contract * quantity * 2  # round trip
        slippage = self.slippage_ticks * self.tick_size * quantity
        spread = self.spread_ticks * self.tick_size * quantity * 0.5  # half spread each way
        impact = price * (self.market_impact_bps / 10000) * quantity

        return commission + slippage + spread + impact

    def entry_cost(self, price: float, quantity: int = 1) -> float:
        """Cost applied at entry (half of round-trip costs excluding commission)."""
        commission = self.commission_per_contract * quantity
        slippage = self.slippage_ticks * self.tick_size * quantity * 0.5
        spread = self.spread_ticks * self.tick_size * quantity * 0.5
        impact = price * (self.market_impact_bps / 10000) * quantity * 0.5

        return commission + slippage + spread + impact

    def exit_cost(self, price: float, quantity: int = 1) -> float:
        """Cost applied at exit."""
        return self.entry_cost(price, quantity)

    def adjusted_entry_price(self, price: float, direction: int, quantity: int = 1) -> float:
        """Get the fill price after costs for an entry.

        Args:
            price: The signal price.
            direction: 1 for long, -1 for short.
            quantity: Number of contracts.

        Returns:
            Adjusted fill price (worse than signal price).
        """
        slip = (self.slippage_ticks + self.spread_ticks * 0.5) * self.tick_size
        impact = price * (self.market_impact_bps / 10000)
        return price + direction * (slip + impact)

    def adjusted_exit_price(self, price: float, direction: int, quantity: int = 1) -> float:
        """Get the fill price after costs for an exit.

        Direction is the position direction (so exit is opposite).
        """
        slip = (self.slippage_ticks + self.spread_ticks * 0.5) * self.tick_size
        impact = price * (self.market_impact_bps / 10000)
        return price - direction * (slip + impact)

    def summary(self) -> dict:
        """Return a summary of cost assumptions."""
        return {
            "commission_per_contract": self.commission_per_contract,
            "slippage_ticks": self.slippage_ticks,
            "tick_size": self.tick_size,
            "spread_ticks": self.spread_ticks,
            "market_impact_bps": self.market_impact_bps,
            "signal_delay_bars": self.signal_delay_bars,
            "estimated_round_trip_cost_1lot_at_5000": round(
                self.total_cost_per_trade(5000.0, 1), 2
            ),
        }


# Preset cost models for common instruments
ES_FUTURES = CostModel(
    commission_per_contract=2.25,
    slippage_ticks=1.0,
    tick_size=0.25,
    spread_ticks=1.0,
    market_impact_bps=0.0,
)

NQ_FUTURES = CostModel(
    commission_per_contract=2.25,
    slippage_ticks=1.0,
    tick_size=0.25,
    spread_ticks=1.0,
    market_impact_bps=0.0,
)

US_EQUITIES = CostModel(
    commission_per_contract=0.005,  # per share
    slippage_ticks=1.0,
    tick_size=0.01,
    spread_ticks=2.0,
    market_impact_bps=5.0,
)
