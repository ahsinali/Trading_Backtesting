"""Slippage and commission models for the backtesting engine."""

from __future__ import annotations

# ── Slippage / commission table ───────────────────────────────────────────────
# Keys are instrument-class strings (passed by the UI or FillEngine).
# Each entry specifies:
#   slippage_ticks          — ticks of adverse fill per side
#   commission_per_share    — flat per-share commission  (equities)
#   commission_per_contract — flat per-contract fee      (futures)
#   commission_per_lot      — flat per-lot fee           (FX)

SLIPPAGE_TABLE: dict[str, dict] = {
    "equity":  {"slippage_ticks": 1, "commission_per_share": 0.005},
    "futures": {"slippage_ticks": 1, "commission_per_contract": 2.50},
    "fx":      {"slippage_ticks": 2, "commission_per_lot": 0.0},
}


def get_slippage(symbol_class: str, tick_size: float) -> float:
    """Return slippage in price units for one fill.

    Parameters
    ----------
    symbol_class:
        Instrument class key (``'equity'``, ``'futures'``, or ``'fx'``).
        Falls back to ``'equity'`` if the key is not found.
    tick_size:
        Minimum price increment for the instrument.
    """
    entry = SLIPPAGE_TABLE.get(symbol_class, SLIPPAGE_TABLE["equity"])
    return entry["slippage_ticks"] * tick_size


def get_commission(symbol_class: str, quantity: float) -> float:
    """Return total commission for a fill of *quantity* units.

    Parameters
    ----------
    symbol_class:
        Instrument class key.
    quantity:
        Number of shares / contracts / lots traded.
    """
    entry = SLIPPAGE_TABLE.get(symbol_class, SLIPPAGE_TABLE["equity"])
    if "commission_per_share" in entry:
        return entry["commission_per_share"] * quantity
    if "commission_per_contract" in entry:
        return entry["commission_per_contract"] * quantity
    if "commission_per_lot" in entry:
        return entry["commission_per_lot"] * quantity
    return 0.0
