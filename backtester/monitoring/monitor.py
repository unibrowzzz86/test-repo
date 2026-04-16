"""Live monitoring framework.

Tracks live strategy performance against backtest expectations.
Raises alerts when performance deviates from what was expected,
including drawdown alerts, win rate degradation, and regime changes.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Alert:
    timestamp: datetime
    level: str  # "info", "warning", "critical"
    category: str
    message: str


class LiveMonitor:
    """Monitor live trading against backtest expectations.

    Compares real-time performance to backtest benchmarks and
    generates alerts when things deviate.
    """

    def __init__(
        self,
        expected_win_rate: float = 55.0,
        expected_profit_factor: float = 1.5,
        max_drawdown_alert_pct: float = 10.0,
        max_drawdown_kill_pct: float = 20.0,
        win_rate_degradation_pct: float = 10.0,
        min_trades_for_alert: int = 20,
        max_consecutive_losses: int = 8,
    ):
        self.expected_win_rate = expected_win_rate
        self.expected_profit_factor = expected_profit_factor
        self.max_drawdown_alert_pct = max_drawdown_alert_pct
        self.max_drawdown_kill_pct = max_drawdown_kill_pct
        self.win_rate_degradation_pct = win_rate_degradation_pct
        self.min_trades_for_alert = min_trades_for_alert
        self.max_consecutive_losses = max_consecutive_losses

        self.trades: list[dict] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self.alerts: list[Alert] = []
        self.peak_equity = 0.0
        self.is_active = True

    def record_trade(self, trade: dict):
        """Record a completed trade and check for alerts.

        Args:
            trade: Dict with keys: timestamp, pnl, direction, entry_price, exit_price.
        """
        self.trades.append(trade)
        self._check_alerts()

    def update_equity(self, timestamp: datetime, equity: float):
        """Update current equity value."""
        self.equity_history.append((timestamp, equity))
        self.peak_equity = max(self.peak_equity, equity)
        self._check_drawdown(timestamp, equity)

    def _check_alerts(self):
        """Run all alert checks after a new trade."""
        if len(self.trades) < self.min_trades_for_alert:
            return

        self._check_win_rate()
        self._check_consecutive_losses()
        self._check_profit_factor()

    def _check_win_rate(self):
        """Alert if win rate has degraded significantly."""
        recent = self.trades[-self.min_trades_for_alert:]
        wins = sum(1 for t in recent if t["pnl"] > 0)
        recent_wr = wins / len(recent) * 100

        if recent_wr < self.expected_win_rate - self.win_rate_degradation_pct:
            self.alerts.append(Alert(
                timestamp=datetime.now(),
                level="warning",
                category="win_rate",
                message=(
                    f"Win rate degraded to {recent_wr:.1f}% "
                    f"(expected {self.expected_win_rate:.1f}%) "
                    f"over last {len(recent)} trades"
                ),
            ))

    def _check_consecutive_losses(self):
        """Alert on excessive consecutive losses."""
        consec = 0
        for t in reversed(self.trades):
            if t["pnl"] < 0:
                consec += 1
            else:
                break

        if consec >= self.max_consecutive_losses:
            self.alerts.append(Alert(
                timestamp=datetime.now(),
                level="critical",
                category="consecutive_losses",
                message=f"{consec} consecutive losses - consider pausing strategy",
            ))

    def _check_profit_factor(self):
        """Alert if profit factor has dropped below threshold."""
        recent = self.trades[-self.min_trades_for_alert:]
        gross_profit = sum(t["pnl"] for t in recent if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in recent if t["pnl"] < 0))

        if gross_loss > 0:
            pf = gross_profit / gross_loss
            if pf < 1.0:
                self.alerts.append(Alert(
                    timestamp=datetime.now(),
                    level="critical",
                    category="profit_factor",
                    message=f"Profit factor dropped to {pf:.2f} over last {len(recent)} trades",
                ))
            elif pf < self.expected_profit_factor * 0.7:
                self.alerts.append(Alert(
                    timestamp=datetime.now(),
                    level="warning",
                    category="profit_factor",
                    message=f"Profit factor at {pf:.2f} (expected {self.expected_profit_factor:.2f})",
                ))

    def _check_drawdown(self, timestamp: datetime, equity: float):
        """Check if drawdown exceeds thresholds."""
        if self.peak_equity <= 0:
            return

        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100

        if dd_pct >= self.max_drawdown_kill_pct:
            self.is_active = False
            self.alerts.append(Alert(
                timestamp=timestamp,
                level="critical",
                category="drawdown_kill",
                message=f"KILL SWITCH: Drawdown at {dd_pct:.1f}% exceeds {self.max_drawdown_kill_pct}% limit. Strategy HALTED.",
            ))
        elif dd_pct >= self.max_drawdown_alert_pct:
            self.alerts.append(Alert(
                timestamp=timestamp,
                level="warning",
                category="drawdown",
                message=f"Drawdown at {dd_pct:.1f}% approaching kill level of {self.max_drawdown_kill_pct}%",
            ))

    def get_status(self) -> dict:
        """Get current monitoring status."""
        if not self.trades:
            return {"status": "no_trades", "is_active": self.is_active}

        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        dd_pct = 0
        if self.peak_equity > 0 and self.equity_history:
            current_eq = self.equity_history[-1][1]
            dd_pct = (self.peak_equity - current_eq) / self.peak_equity * 100

        return {
            "is_active": self.is_active,
            "total_trades": len(self.trades),
            "total_pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / len(self.trades) * 100, 1),
            "current_drawdown_pct": round(dd_pct, 2),
            "num_alerts": len(self.alerts),
            "recent_alerts": [
                {"level": a.level, "category": a.category, "message": a.message}
                for a in self.alerts[-5:]
            ],
        }

    def get_checklist(self) -> dict:
        """Generate a go-live readiness checklist."""
        return {
            "evidence_edge_is_real": {
                "out_of_sample_tested": "Run walk-forward analysis",
                "walk_forward_efficiency": "Check WF efficiency > 50%",
                "performance_by_regime": "Run regime analysis",
                "enough_trades": f"Need 100+ trades (have {len(self.trades)})",
                "not_dependent_on_lucky_periods": "Check Monte Carlo results",
            },
            "realistic_costs": {
                "commissions_modeled": "Verify cost model matches broker",
                "slippage_modeled": "Verify slippage assumptions",
                "spread_modeled": "Check spread assumptions",
                "signal_delay": "Add realistic signal-to-fill delay",
            },
            "risk_profile": {
                "max_drawdown_acceptable": f"Max DD limit: {self.max_drawdown_kill_pct}%",
                "worst_month_survivable": "Check worst month in backtest",
                "loss_clusters_handled": "Check worst 5-trade sequence",
            },
            "position_sizing": {
                "capital_per_trade_defined": "Set max % per trade",
                "daily_loss_limit_set": "Set daily loss limit",
                "drawdown_kill_switch": f"Kill at {self.max_drawdown_kill_pct}% DD",
            },
            "implementation": {
                "data_feed_reliable": "Verify data source and latency",
                "order_routing_tested": "Paper trade first",
                "no_look_ahead_bias": "Verify no future data leakage",
                "corporate_actions_handled": "N/A for ES futures",
            },
            "monitoring": {
                "win_rate_tracking": f"Alert if WR drops {self.win_rate_degradation_pct}% below {self.expected_win_rate}%",
                "drawdown_alerts": f"Alert at {self.max_drawdown_alert_pct}% DD",
                "consecutive_loss_limit": f"Alert at {self.max_consecutive_losses} consecutive losses",
            },
        }
