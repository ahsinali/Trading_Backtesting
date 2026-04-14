"""Trade log analytics — equity curve and summary statistics."""

from __future__ import annotations

import pandas as pd

from backtester.sim.orders import Trade


def compute_equity_curve(trades: list[Trade]) -> pd.Series:
    """Return the cumulative P&L series for *trades*.

    Parameters
    ----------
    trades:
        Completed trades in chronological order.

    Returns
    -------
    pd.Series
        Integer-indexed (1-based trade number) cumulative P&L.
        Empty series if *trades* is empty.
    """
    if not trades:
        return pd.Series(dtype=float)
    pnls = [t.pnl_currency for t in trades]
    return pd.Series(pnls, index=range(1, len(pnls) + 1)).cumsum()


def compute_summary_stats(trades: list[Trade]) -> dict:
    """Return a dict of session-level metrics for *trades*.

    Keys
    ----
    net_pnl, avg_trade_pnl, win_rate, profit_factor,
    expectancy_currency, expectancy_r, max_drawdown, trade_count.

    All values are ``0.0`` (or ``0`` for *trade_count*) when *trades* is empty.
    ``profit_factor`` is ``float('inf')`` when there are no losing trades.
    """
    if not trades:
        return {
            "net_pnl":            0.0,
            "avg_trade_pnl":      0.0,
            "win_rate":           0.0,
            "profit_factor":      0.0,
            "expectancy_currency": 0.0,
            "expectancy_r":       0.0,
            "max_drawdown":       0.0,
            "trade_count":        0,
        }

    pnls   = [t.pnl_currency for t in trades]
    r_vals = [t.pnl_r for t in trades if t.pnl_r is not None]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    net_pnl       = sum(pnls)
    avg_trade_pnl = net_pnl / len(trades)
    win_rate      = len(wins) / len(trades)
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy_r  = sum(r_vals) / len(r_vals) if r_vals else 0.0

    # Max drawdown from cumulative equity curve
    equity   = pd.Series(pnls).cumsum()
    peak     = equity.cummax()
    drawdown = equity - peak
    max_dd   = float(drawdown.min())

    return {
        "net_pnl":            net_pnl,
        "avg_trade_pnl":      avg_trade_pnl,
        "win_rate":           win_rate,
        "profit_factor":      profit_factor,
        "expectancy_currency": avg_trade_pnl,
        "expectancy_r":       expectancy_r,
        "max_drawdown":       max_dd,
        "trade_count":        len(trades),
    }
