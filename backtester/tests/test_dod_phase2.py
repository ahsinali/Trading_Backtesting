"""Phase 2 Definition-of-Done assertions.

These tests are the canonical acceptance criteria for Phase 2:
  - OCO invariant: closing one bracket leg removes the position entirely
  - Ambiguity policy consistency: each policy resolves predictably
  - Trade log math: P&L and R spot-checks for known entry/exit combos
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.sim.fills import FillEngine
from backtester.sim.orders import Order
from backtester.sim.pnl import compute_pnl_currency, compute_pnl_r


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(
    open_: float, high: float, low: float, close: float,
    date: str = "2020-01-03",
) -> pd.Series:
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close},
        name=pd.Timestamp(date),
    )


def _market(
    side: str = "long",
    stop: float | None = None,
    target: float | None = None,
    qty: float = 1.0,
    tick_size: float = 0.01,
) -> Order:
    return Order.market(
        session_id       = "dod-sess",
        side             = side,
        quantity         = qty,
        tick_size        = tick_size,
        created_at_index = 0,
        stop_price       = stop,
        target_price     = target,
    )


def _engine(rule: str = "default", tick_size: float = 0.01) -> FillEngine:
    return FillEngine(tick_size=tick_size, execution_rule=rule)


# ── DoD 1: OCO invariant ──────────────────────────────────────────────────────

def test_oco_invariant() -> None:
    """After a bracket exit, exactly 0 open positions remain.

    Once either the stop or the target fires, the position must be removed
    from open_positions so the other leg can never trigger — the OCO guarantee.
    """
    engine = _engine()
    orders = [_market("long", stop=90.0, target=115.0)]

    # Bar 1: entry fills at open=100
    engine.process_bar(_bar(100, 102, 99, 101, "2020-01-01"), orders)
    assert len(engine.open_positions) == 1

    # Bar 2: high=120 → target hit; low=107 → stop NOT hit
    trades = engine.process_bar(_bar(108, 120, 107, 116, "2020-01-02"), [])

    assert len(trades) == 1, "Expected exactly one completed trade"
    assert trades[0].exit_reason == "target"
    assert len(engine.open_positions) == 0, (
        "Position must be removed after target fills — stop leg must not remain"
    )

    # Bar 3: confirm stop leg is truly gone (no phantom fill)
    ghost_trades = engine.process_bar(_bar(80, 82, 78, 79, "2020-01-03"), [])
    assert len(ghost_trades) == 0, "Phantom stop fill on bar 3 — OCO violated"


def test_oco_stop_leg() -> None:
    """Mirror test: stop fires first → position gone, target never fills."""
    engine = _engine()
    orders = [_market("long", stop=90.0, target=115.0)]

    engine.process_bar(_bar(100, 102, 99, 101, "2020-01-01"), orders)

    # Bar 2: low=85 → stop hit; high=112 → target NOT hit
    trades = engine.process_bar(_bar(95, 112, 85, 87, "2020-01-02"), [])
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert len(engine.open_positions) == 0

    # Bar 3: target would have been reachable, but no open position
    ghost = engine.process_bar(_bar(108, 120, 107, 116, "2020-01-03"), [])
    assert len(ghost) == 0


# ── DoD 2: Ambiguity policy consistency ───────────────────────────────────────

def test_ambiguity_policy_consistency() -> None:
    """Each ambiguity policy resolves consistently on a bar that hits both legs.

    Bar conditions: open>close (bearish bar), both stop=90 and target=115 hit.
    Expected outcomes:
      - default      → stop (bearish bar → stop-first path)
      - conservative → stop
      - optimistic   → target
      - unknown      → no trade on this bar (deferred)
    """
    def _run(policy: str) -> list:
        engine = _engine(policy)
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100, 102, 99, 101, "2020-01-01"), orders)
        # Bearish bar that touches both legs
        return engine.process_bar(_bar(108, 120, 85, 86, "2020-01-02"), [])

    cases = [
        ("default",      "stop"),
        ("conservative", "stop"),
        ("optimistic",   "target"),
        ("unknown",      None),
    ]
    for policy, expected_reason in cases:
        trades = _run(policy)
        if expected_reason is None:
            assert len(trades) == 0, (
                f"Policy '{policy}': expected deferred (no trade), got {len(trades)} trade(s)"
            )
        else:
            assert len(trades) == 1, (
                f"Policy '{policy}': expected 1 trade, got {len(trades)}"
            )
            assert trades[0].exit_reason == expected_reason, (
                f"Policy '{policy}': expected exit_reason='{expected_reason}', "
                f"got '{trades[0].exit_reason}'"
            )


def test_ambiguity_unknown_resolves_next_bar() -> None:
    """'unknown' policy defers on bar 2 then resolves on bar 3 with flag=1."""
    engine = _engine("unknown")
    orders = [_market("long", stop=90.0, target=115.0)]
    engine.process_bar(_bar(100, 102, 99, 101, "2020-01-01"), orders)

    # Bar 2: both legs hit → deferred
    deferred = engine.process_bar(_bar(108, 120, 85, 86, "2020-01-02"), [])
    assert len(deferred) == 0
    assert len(engine.open_positions) == 1
    assert engine.open_positions[0].ambiguous is True

    # Bar 3: only target hit → resolves; ambiguity_flag must be 1
    resolved = engine.process_bar(_bar(108, 120, 107, 116, "2020-01-03"), [])
    assert len(resolved) == 1
    assert resolved[0].exit_reason == "target"
    assert resolved[0].ambiguity_flag == 1, (
        "Deferred position must carry ambiguity_flag=1 when it eventually closes"
    )


# ── DoD 3: Trade log math ─────────────────────────────────────────────────────

@pytest.mark.parametrize("entry,exit_,stop,side,qty,exp_pnl,exp_r", [
    # 1R winner: entry=100, exit=110, stop=90, long, qty=1 → PnL=10, R=1.0
    (100.0, 110.0,  90.0, "long",  1.0,  10.0,  1.0),
    # 0.8R loser: entry=100, exit=92, stop=90, long, qty=1 → PnL=-8, R=-0.8
    (100.0,  92.0,  90.0, "long",  1.0,  -8.0, -0.8),
    # Short winner: entry=100, exit=80, stop=110, short, qty=2 → PnL=40, R=2.0
    (100.0,  80.0, 110.0, "short", 2.0,  40.0,  2.0),
    # No stop: entry=100, exit=80, no stop, long, qty=2 → PnL=-40, R=None
    (100.0,  80.0,  None, "long",  2.0, -40.0,  None),
])
def test_trade_log_math(
    entry: float, exit_: float, stop: float | None,
    side: str, qty: float,
    exp_pnl: float, exp_r: float | None,
) -> None:
    """Spot-check P&L and R calculations for known entry/exit/stop combos."""
    pnl = compute_pnl_currency(entry, exit_, side, qty, commission=0.0, slippage=0.0)
    r   = compute_pnl_r(entry, exit_, stop, side, qty, commission=0.0, slippage=0.0)

    assert pnl == pytest.approx(exp_pnl), (
        f"PnL mismatch: entry={entry} exit={exit_} side={side} qty={qty} "
        f"→ expected {exp_pnl}, got {pnl}"
    )
    if exp_r is None:
        assert r is None, f"Expected R=None (no stop), got {r}"
    else:
        assert r == pytest.approx(exp_r), (
            f"R mismatch: entry={entry} exit={exit_} stop={stop} side={side} "
            f"→ expected {exp_r}, got {r}"
        )
