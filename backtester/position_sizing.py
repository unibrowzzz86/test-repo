"""Position sizing and kill rules.

Controls how much capital goes into each trade and when
to reduce or shut down the strategy entirely.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class PositionSizer:
    """Position sizing with risk management kill rules.

    Attributes:
        max_position_size: Maximum contracts/shares per trade.
        max_capital_per_trade_pct: Max % of equity per trade.
        max_gross_exposure_pct: Max gross exposure as % of equity.
        max_net_exposure_pct: Max net exposure as % of equity.
        max_sector_concentration_pct: Max % in any single sector/asset.
        daily_loss_limit_pct: Daily loss limit as % of equity (kills trading for the day).
        drawdown_reduce_pct: Drawdown % at which to reduce position size.
        drawdown_stop_pct: Drawdown % at which to stop trading entirely.
        reduce_size_factor: Factor to multiply position size when in reduce mode.
    """

    max_position_size: int = 1
    max_capital_per_trade_pct: float = 2.0
    max_gross_exposure_pct: float = 100.0
    max_net_exposure_pct: float = 100.0
    max_sector_concentration_pct: float = 25.0
    daily_loss_limit_pct: float = 3.0
    drawdown_reduce_pct: float = 10.0
    drawdown_stop_pct: float = 20.0
    reduce_size_factor: float = 0.5

    def calculate_size(
        self,
        equity: float,
        price: float,
        risk_per_unit: float,
        current_drawdown_pct: float = 0.0,
        daily_pnl_pct: float = 0.0,
    ) -> int:
        """Calculate position size based on risk parameters.

        Args:
            equity: Current account equity.
            price: Current price of the instrument.
            risk_per_unit: Dollar risk per contract/share (e.g., stop distance * tick value).
            current_drawdown_pct: Current drawdown from peak as a percentage.
            daily_pnl_pct: Today's P&L as percentage of equity.

        Returns:
            Number of contracts/shares to trade (0 if killed).
        """
        if self.should_stop(current_drawdown_pct, daily_pnl_pct):
            return 0

        max_risk = equity * (self.max_capital_per_trade_pct / 100)

        if risk_per_unit > 0:
            size = int(max_risk / risk_per_unit)
        else:
            size = self.max_position_size

        size = min(size, self.max_position_size)

        if self.should_reduce(current_drawdown_pct):
            size = max(1, int(size * self.reduce_size_factor))

        return max(0, size)

    def should_reduce(self, current_drawdown_pct: float) -> bool:
        """Check if position size should be reduced."""
        return current_drawdown_pct >= self.drawdown_reduce_pct

    def should_stop(self, current_drawdown_pct: float, daily_pnl_pct: float = 0.0) -> bool:
        """Check if trading should be stopped entirely."""
        if current_drawdown_pct >= self.drawdown_stop_pct:
            return True
        if abs(daily_pnl_pct) >= self.daily_loss_limit_pct and daily_pnl_pct < 0:
            return True
        return False

    def check_exposure(
        self,
        positions: dict,
        equity: float,
    ) -> dict:
        """Check current exposure against limits.

        Args:
            positions: Dict of {symbol: (quantity, price, sector)}.
            equity: Current account equity.

        Returns:
            Dict with exposure metrics and whether limits are breached.
        """
        gross_exposure = 0.0
        net_exposure = 0.0
        sector_exposure: dict[str, float] = {}

        for symbol, (qty, price, sector) in positions.items():
            notional = abs(qty) * price
            gross_exposure += notional
            net_exposure += qty * price

            if sector not in sector_exposure:
                sector_exposure[sector] = 0.0
            sector_exposure[sector] += notional

        gross_pct = (gross_exposure / equity * 100) if equity > 0 else 0
        net_pct = (abs(net_exposure) / equity * 100) if equity > 0 else 0

        sector_pcts = {s: (v / equity * 100) if equity > 0 else 0 for s, v in sector_exposure.items()}
        max_sector_pct = max(sector_pcts.values()) if sector_pcts else 0

        return {
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "gross_exposure_pct": round(gross_pct, 2),
            "net_exposure_pct": round(net_pct, 2),
            "sector_exposure_pct": sector_pcts,
            "max_sector_pct": round(max_sector_pct, 2),
            "gross_limit_breached": gross_pct > self.max_gross_exposure_pct,
            "net_limit_breached": net_pct > self.max_net_exposure_pct,
            "sector_limit_breached": max_sector_pct > self.max_sector_concentration_pct,
        }

    def summary(self) -> dict:
        """Return a summary of position sizing rules."""
        return {
            "max_position_size": self.max_position_size,
            "max_capital_per_trade_pct": self.max_capital_per_trade_pct,
            "max_gross_exposure_pct": self.max_gross_exposure_pct,
            "max_net_exposure_pct": self.max_net_exposure_pct,
            "max_sector_concentration_pct": self.max_sector_concentration_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "drawdown_reduce_pct": self.drawdown_reduce_pct,
            "drawdown_stop_pct": self.drawdown_stop_pct,
            "reduce_size_factor": self.reduce_size_factor,
        }
