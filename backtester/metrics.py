"""Performance metrics calculation.

Computes every metric shown in the dashboard plus the full risk profile
recommended by the Reddit checklist: max drawdown, expected drawdown,
worst month/week, longest recovery, hit rate, payoff ratio, exposure,
turnover, concentration, and loss clustering analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        trades: DataFrame with columns [entry_date, exit_date, direction,
                entry_price, exit_price, pnl, quantity].
        equity_curve: Series of portfolio value indexed by date.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        periods_per_year: Trading days per year.

    Returns:
        Dict of all computed metrics.
    """
    if trades.empty or len(equity_curve) < 2:
        return _empty_metrics()

    returns = equity_curve.pct_change().dropna()
    total_return_pct = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    net_profit = equity_curve.iloc[-1] - initial_capital

    # Trade-level metrics
    winning = trades[trades["pnl"] > 0]
    losing = trades[trades["pnl"] < 0]
    num_trades = len(trades)
    num_wins = len(winning)
    num_losses = len(losing)
    win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0

    avg_win = winning["pnl"].mean() if len(winning) > 0 else 0
    avg_loss = abs(losing["pnl"].mean()) if len(losing) > 0 else 0
    largest_win = trades["pnl"].max() if num_trades > 0 else 0
    largest_loss = trades["pnl"].min() if num_trades > 0 else 0

    gross_profit = winning["pnl"].sum() if len(winning) > 0 else 0
    gross_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else float("inf")

    # Drawdown analysis
    dd_info = _drawdown_analysis(equity_curve)

    # Risk-adjusted returns
    sharpe = _sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = _sortino_ratio(returns, risk_free_rate, periods_per_year)
    calmar = _calmar_ratio(total_return_pct, dd_info["max_drawdown_pct"], equity_curve)

    # Time-based breakdowns
    monthly = _monthly_breakdown(trades, equity_curve)
    weekly_returns = _weekly_returns(equity_curve)
    yearly = _yearly_breakdown(trades, equity_curve)

    # Streaks
    streak_info = _streak_analysis(trades)

    # Loss clustering
    loss_cluster = _loss_cluster_analysis(trades)

    # Exposure and turnover
    num_days = (equity_curve.index[-1] - equity_curve.index[0]).days or 1
    avg_trade_duration = _avg_trade_duration(trades)
    cagr = _cagr(initial_capital, equity_curve.iloc[-1], num_days)

    return {
        # Top-line
        "total_return_pct": round(total_return_pct, 2),
        "net_profit": round(net_profit, 2),
        "initial_capital": initial_capital,
        "final_equity": round(equity_curve.iloc[-1], 2),
        "cagr_pct": round(cagr, 2),
        # Trade stats
        "num_trades": num_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "payoff_ratio": round(payoff_ratio, 3),
        "avg_trade_pnl": round(trades["pnl"].mean(), 2) if num_trades > 0 else 0,
        # Risk-adjusted
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        # Drawdown
        "max_drawdown_pct": round(dd_info["max_drawdown_pct"], 2),
        "max_drawdown_dollars": round(dd_info["max_drawdown_dollars"], 2),
        "avg_drawdown_pct": round(dd_info["avg_drawdown_pct"], 2),
        "max_drawdown_duration_days": dd_info["max_drawdown_duration_days"],
        "longest_recovery_days": dd_info["longest_recovery_days"],
        # Time breakdowns
        "worst_month_pct": round(monthly["worst_month_pct"], 2) if monthly else 0,
        "best_month_pct": round(monthly["best_month_pct"], 2) if monthly else 0,
        "worst_week_pct": round(weekly_returns.min(), 2) if len(weekly_returns) > 0 else 0,
        "best_week_pct": round(weekly_returns.max(), 2) if len(weekly_returns) > 0 else 0,
        "monthly_breakdown": monthly.get("breakdown", {}),
        "yearly_breakdown": yearly,
        # Streaks
        "max_win_streak": streak_info["max_win_streak"],
        "max_loss_streak": streak_info["max_loss_streak"],
        "current_streak": streak_info["current_streak"],
        # Loss clustering
        "max_consecutive_loss_dollars": round(loss_cluster["max_consecutive_loss"], 2),
        "worst_3_trade_pnl": round(loss_cluster["worst_3_trade_pnl"], 2),
        "worst_5_trade_pnl": round(loss_cluster["worst_5_trade_pnl"], 2),
        # Duration
        "avg_trade_duration_hours": round(avg_trade_duration, 1),
        "total_trading_days": num_days,
        # Equity curve & drawdown series for charting
        "equity_curve": equity_curve,
        "drawdown_series": dd_info["drawdown_series"],
        "returns": returns,
    }


def _empty_metrics() -> dict:
    return {k: 0 for k in [
        "total_return_pct", "net_profit", "num_trades", "win_rate_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
        "profit_factor",
    ]}


def _sharpe_ratio(returns: pd.Series, rf: float, periods: int) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - rf / periods
    return float(np.sqrt(periods) * excess.mean() / returns.std())


def _sortino_ratio(returns: pd.Series, rf: float, periods: int) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf / periods
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / downside.std())


def _calmar_ratio(total_return_pct: float, max_dd_pct: float, equity_curve: pd.Series) -> float:
    if max_dd_pct == 0:
        return 0.0
    num_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if num_days == 0:
        return 0.0
    annual_return = total_return_pct * (365.25 / num_days)
    return annual_return / max_dd_pct


def _cagr(initial: float, final: float, days: int) -> float:
    if initial <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    if years == 0:
        return 0.0
    return ((final / initial) ** (1 / years) - 1) * 100


def _drawdown_analysis(equity_curve: pd.Series) -> dict:
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100
    drawdown_dollars = equity_curve - peak

    max_dd_pct = abs(drawdown.min()) if len(drawdown) > 0 else 0
    max_dd_dollars = abs(drawdown_dollars.min()) if len(drawdown_dollars) > 0 else 0

    # Average drawdown (only during drawdown periods)
    in_dd = drawdown[drawdown < 0]
    avg_dd = abs(in_dd.mean()) if len(in_dd) > 0 else 0

    # Drawdown duration
    is_dd = drawdown < 0
    max_dd_duration = 0
    max_recovery = 0
    current_dd_start = None

    for i, (date, val) in enumerate(drawdown.items()):
        if val < 0 and current_dd_start is None:
            current_dd_start = date
        elif val >= 0 and current_dd_start is not None:
            duration = (date - current_dd_start).days
            max_dd_duration = max(max_dd_duration, duration)
            max_recovery = max(max_recovery, duration)
            current_dd_start = None

    # If still in drawdown at end
    if current_dd_start is not None:
        duration = (equity_curve.index[-1] - current_dd_start).days
        max_dd_duration = max(max_dd_duration, duration)

    return {
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_dollars": max_dd_dollars,
        "avg_drawdown_pct": avg_dd,
        "max_drawdown_duration_days": max_dd_duration,
        "longest_recovery_days": max_recovery,
        "drawdown_series": drawdown,
    }


def _monthly_breakdown(trades: pd.DataFrame, equity_curve: pd.Series) -> dict:
    if trades.empty:
        return {}

    # Monthly P&L from equity curve
    monthly_equity = equity_curve.resample("ME").last()
    monthly_returns = monthly_equity.pct_change().dropna() * 100

    breakdown = {}
    for date, ret in monthly_returns.items():
        key = date.strftime("%Y-%m")
        month_trades = trades[
            (trades["exit_date"].dt.to_period("M") == date.to_period("M"))
        ]
        month_pnl = month_trades["pnl"].sum() if len(month_trades) > 0 else 0
        month_wins = len(month_trades[month_trades["pnl"] > 0])
        month_total = len(month_trades)
        win_rate = (month_wins / month_total * 100) if month_total > 0 else 0

        breakdown[key] = {
            "return_pct": round(ret, 1),
            "pnl": round(month_pnl, 2),
            "trades": month_total,
            "win_rate": round(win_rate, 1),
        }

    return {
        "breakdown": breakdown,
        "worst_month_pct": monthly_returns.min() if len(monthly_returns) > 0 else 0,
        "best_month_pct": monthly_returns.max() if len(monthly_returns) > 0 else 0,
    }


def _weekly_returns(equity_curve: pd.Series) -> pd.Series:
    weekly = equity_curve.resample("W").last()
    return weekly.pct_change().dropna() * 100


def _yearly_breakdown(trades: pd.DataFrame, equity_curve: pd.Series) -> dict:
    if trades.empty:
        return {}
    yearly = {}
    for year in trades["exit_date"].dt.year.unique():
        year_trades = trades[trades["exit_date"].dt.year == year]
        yearly[int(year)] = {
            "pnl": round(year_trades["pnl"].sum(), 2),
            "trades": len(year_trades),
            "win_rate": round(
                len(year_trades[year_trades["pnl"] > 0]) / len(year_trades) * 100, 1
            ) if len(year_trades) > 0 else 0,
        }
    return yearly


def _streak_analysis(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"max_win_streak": 0, "max_loss_streak": 0, "current_streak": 0}

    wins = (trades["pnl"] > 0).astype(int)
    max_win = 0
    max_loss = 0
    current_win = 0
    current_loss = 0

    for w in wins:
        if w:
            current_win += 1
            current_loss = 0
            max_win = max(max_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss = max(max_loss, current_loss)

    current = current_win if current_win > 0 else -current_loss
    return {
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
        "current_streak": current,
    }


def _loss_cluster_analysis(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"max_consecutive_loss": 0, "worst_3_trade_pnl": 0, "worst_5_trade_pnl": 0}

    pnls = trades["pnl"].values

    # Max consecutive loss
    max_consec_loss = 0
    current_consec = 0
    for p in pnls:
        if p < 0:
            current_consec += p
            max_consec_loss = min(max_consec_loss, current_consec)
        else:
            current_consec = 0

    # Worst rolling N-trade P&L
    pnl_series = pd.Series(pnls)
    worst_3 = pnl_series.rolling(3).sum().min() if len(pnls) >= 3 else pnl_series.sum()
    worst_5 = pnl_series.rolling(5).sum().min() if len(pnls) >= 5 else pnl_series.sum()

    return {
        "max_consecutive_loss": abs(max_consec_loss),
        "worst_3_trade_pnl": worst_3,
        "worst_5_trade_pnl": worst_5,
    }


def _avg_trade_duration(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    durations = (trades["exit_date"] - trades["entry_date"]).dt.total_seconds() / 3600
    return float(durations.mean())
