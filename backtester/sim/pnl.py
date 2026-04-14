"""P&L calculation helpers."""

from __future__ import annotations


def compute_pnl_currency(
    entry:      float,
    exit_price: float,
    side:       str,
    quantity:   float,
    commission: float = 0.0,
    slippage:   float = 0.0,
) -> float:
    """Net P&L in currency units.

    Parameters
    ----------
    entry:
        Entry fill price.
    exit_price:
        Exit fill price.
    side:
        ``'long'`` or ``'short'``.
    quantity:
        Number of units traded.
    commission:
        Total commission cost (flat, not per-unit).
    slippage:
        Total slippage cost (flat, not per-unit).

    Returns
    -------
    float
        ``(exit − entry) × quantity − commission − slippage`` for longs;
        ``(entry − exit) × quantity − commission − slippage`` for shorts.
    """
    if side == "long":
        gross = (exit_price - entry) * quantity
    else:
        gross = (entry - exit_price) * quantity
    return gross - commission - slippage


def compute_pnl_r(
    entry:      float,
    exit_price: float,
    stop:       float | None,
    side:       str,
    quantity:   float,
    commission: float = 0.0,
    slippage:   float = 0.0,
) -> float | None:
    """Net P&L expressed as a multiple of the initial risk (R).

    Returns ``None`` when *stop* is ``None`` (risk undefined) or when the
    computed risk is zero / negative (invalid stop placement).

    Formula
    -------
    ``R = net_pnl / risk``

    where ``risk = (entry − stop) × quantity`` for longs and
    ``(stop − entry) × quantity`` for shorts.
    """
    if stop is None:
        return None

    net_pnl = compute_pnl_currency(entry, exit_price, side, quantity, commission, slippage)

    if side == "long":
        risk = (entry - stop) * quantity
    else:
        risk = (stop - entry) * quantity

    if risk <= 0:
        return None  # stop on wrong side of entry — undefined R

    return net_pnl / risk
