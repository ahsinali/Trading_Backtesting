"""Tests for FillEngine, P&L helpers, OCO behaviour, and tick snapping."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.sim.fills import FillEngine, flatten_position
from backtester.sim.orders import Order, Trade, snap_to_tick
from backtester.sim.pnl import compute_pnl_currency, compute_pnl_r


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(open_: float, high: float, low: float, close: float,
         date: str = "2020-01-01") -> pd.Series:
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close},
        name=pd.Timestamp(date),
    )


def _market(side: str = "long", qty: float = 1.0,
            stop: float | None = None, target: float | None = None,
            tick_size: float = 0.01) -> Order:
    return Order.market(
        session_id       = "test-sess",
        side             = side,
        quantity         = qty,
        tick_size        = tick_size,
        created_at_index = 0,
        stop_price       = stop,
        target_price     = target,
    )


# ── snap_to_tick ──────────────────────────────────────────────────────────────

class TestSnapToTick:
    def test_exact_multiple(self):
        assert snap_to_tick(1.00, 0.01) == pytest.approx(1.00)

    def test_rounds_up(self):
        assert snap_to_tick(1.005, 0.01) == pytest.approx(1.01)

    def test_rounds_down(self):
        assert snap_to_tick(1.004, 0.01) == pytest.approx(1.00)

    def test_zero_tick_passthrough(self):
        assert snap_to_tick(1.23456, 0) == pytest.approx(1.23456)

    def test_coarse_tick(self):
        assert snap_to_tick(1.03, 0.05) == pytest.approx(1.05)


# ── Market fills ──────────────────────────────────────────────────────────────

class TestMarketFills:
    def test_fills_at_next_bar_open(self):
        """Market order fills at bar.open; no trade returned on entry bar."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long")]
        bar    = _bar(105.0, 108.0, 104.0, 107.0)
        trades = engine.process_bar(bar, orders)
        assert orders[0].status == "filled"
        assert len(trades) == 0                         # no exit yet
        assert len(engine.open_positions) == 1
        assert engine.open_positions[0].entry_price == pytest.approx(105.0)

    def test_entry_price_snapped_to_tick(self):
        engine = FillEngine(tick_size=0.05)
        orders = [_market("long", tick_size=0.05)]
        bar    = _bar(100.03, 101.0, 99.0, 100.5)      # open not on tick
        engine.process_bar(bar, orders)
        assert engine.open_positions[0].entry_price == pytest.approx(100.05)

    def test_already_filled_order_skipped(self):
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long")]
        bar    = _bar(100.0, 102.0, 99.0, 101.0)
        engine.process_bar(bar, orders)
        assert orders[0].status == "filled"
        # Process same order again — should not open a second position
        engine.process_bar(bar, orders)
        assert len(engine.open_positions) == 1

    def test_short_market_fills_at_open(self):
        engine = FillEngine(tick_size=0.01)
        orders = [_market("short")]
        bar    = _bar(200.0, 201.0, 199.0, 200.5)
        engine.process_bar(bar, orders)
        assert engine.open_positions[0].entry_price == pytest.approx(200.0)
        assert engine.open_positions[0].side == "short"


# ── Stop / target exit ────────────────────────────────────────────────────────

class TestExits:
    def test_stop_triggered_for_long(self):
        """Long position stops out when bar low ≤ stop_price."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", stop=90.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)

        trades = engine.process_bar(_bar(91.0, 92.0, 85.0, 89.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"
        assert trades[0].exit_price  == pytest.approx(90.0)

    def test_target_triggered_for_long(self):
        """Long position hits target when bar high ≥ target_price."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)

        trades = engine.process_bar(_bar(108.0, 120.0, 107.0, 116.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "target"
        assert trades[0].exit_price  == pytest.approx(115.0)

    def test_stop_triggered_for_short(self):
        engine = FillEngine(tick_size=0.01)
        orders = [_market("short", stop=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)

        trades = engine.process_bar(_bar(108.0, 120.0, 107.0, 118.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"

    def test_no_exit_when_neither_touched(self):
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)

        trades = engine.process_bar(_bar(100.0, 110.0, 95.0, 105.0, "2020-01-02"), [])
        assert len(trades) == 0
        assert len(engine.open_positions) == 1


# ── OCO behaviour ─────────────────────────────────────────────────────────────

class TestOCO:
    def test_stop_fill_cancels_target(self):
        """After stop triggers, target can no longer fill (position is gone)."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 98.0, 101.0, "2020-01-01"), orders)

        # Stop hit on bar 2
        trades = engine.process_bar(_bar(91.0, 93.0, 85.0, 89.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"

        # Bar 3 would have hit the target — but position is already closed
        trades = engine.process_bar(_bar(89.0, 120.0, 88.0, 118.0, "2020-01-03"), [])
        assert len(trades) == 0
        assert len(engine.open_positions) == 0

    def test_target_fill_cancels_stop(self):
        """After target triggers, stop can no longer fill."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 98.0, 101.0, "2020-01-01"), orders)

        # Target hit on bar 2
        trades = engine.process_bar(_bar(108.0, 120.0, 107.0, 118.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "target"

        # Bar 3 goes down to stop level — but position is already closed
        trades = engine.process_bar(_bar(80.0, 82.0, 78.0, 81.0, "2020-01-03"), [])
        assert len(trades) == 0
        assert len(engine.open_positions) == 0

    def test_multiple_open_positions_independent(self):
        """Two open positions close independently; one stop does not affect the other."""
        engine = FillEngine(tick_size=0.01)
        o1 = _market("long", stop=90.0, target=120.0)
        o2 = _market("long", stop=95.0, target=130.0)
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), [o1, o2])
        assert len(engine.open_positions) == 2

        # o2 stop=95 is hit (low=92), o1 stop=90 is not (low=92 > 90)
        trades = engine.process_bar(_bar(98.0, 99.0, 92.0, 96.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].stop_price == pytest.approx(95.0)
        assert len(engine.open_positions) == 1


# ── Ambiguity (default policy) ────────────────────────────────────────────────

class TestAmbiguity:
    def test_bullish_bar_target_wins(self):
        """Default policy: bullish bar → High before Low → target fills first."""
        engine = FillEngine(tick_size=0.01, execution_rule="default")
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)

        # Both touched; close(120) > open(108) → bullish
        trades = engine.process_bar(_bar(108.0, 120.0, 85.0, 120.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "target"
        assert trades[0].ambiguity_flag == 0

    def test_bearish_bar_stop_wins(self):
        """Default policy: bearish bar → Low before High → stop fills first."""
        engine = FillEngine(tick_size=0.01, execution_rule="default")
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)

        # Both touched; close(86) < open(108) → bearish
        trades = engine.process_bar(_bar(108.0, 120.0, 85.0, 86.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop"

    def test_conservative_always_stop(self):
        engine = FillEngine(tick_size=0.01, execution_rule="conservative")
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)
        trades = engine.process_bar(_bar(108.0, 120.0, 85.0, 120.0, "2020-01-02"), [])
        assert trades[0].exit_reason == "stop"

    def test_optimistic_always_target(self):
        engine = FillEngine(tick_size=0.01, execution_rule="optimistic")
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)
        trades = engine.process_bar(_bar(108.0, 120.0, 85.0, 86.0, "2020-01-02"), [])
        assert trades[0].exit_reason == "target"

    def test_unknown_defers_exit(self):
        # Phase 2B: 'unknown' policy does NOT close the position on the ambiguous bar.
        engine = FillEngine(tick_size=0.01, execution_rule="unknown")
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)
        trades = engine.process_bar(_bar(108.0, 120.0, 85.0, 86.0, "2020-01-02"), [])
        assert len(trades) == 0
        assert len(engine.open_positions) == 1
        assert engine.open_positions[0].ambiguous is True

    def test_unknown_resolves_next_bar(self):
        # After deferral, position closes on the next bar that hits stop or target.
        engine = FillEngine(tick_size=0.01, execution_rule="unknown")
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 102.0, 99.0, 101.0, "2020-01-01"), orders)
        engine.process_bar(_bar(108.0, 120.0, 85.0, 86.0, "2020-01-02"), [])
        # Bar 3: only target hit → closes; flag=1 because it was previously ambiguous
        trades = engine.process_bar(_bar(112.0, 120.0, 113.0, 118.0, "2020-01-03"), [])
        assert len(trades) == 1
        assert trades[0].exit_reason == "target"
        assert trades[0].ambiguity_flag == 1


# ── P&L math ──────────────────────────────────────────────────────────────────

class TestPnlCurrency:
    def test_long_profit(self):
        """Long entry 100, exit 110, qty 10 → pnl = 100.0"""
        pnl = compute_pnl_currency(100.0, 110.0, "long", 10.0, 0.0, 0.0)
        assert pnl == pytest.approx(100.0)

    def test_long_loss(self):
        pnl = compute_pnl_currency(100.0, 90.0, "long", 1.0, 0.0, 0.0)
        assert pnl == pytest.approx(-10.0)

    def test_short_profit(self):
        pnl = compute_pnl_currency(110.0, 100.0, "short", 1.0, 0.0, 0.0)
        assert pnl == pytest.approx(10.0)

    def test_short_loss(self):
        pnl = compute_pnl_currency(100.0, 110.0, "short", 1.0, 0.0, 0.0)
        assert pnl == pytest.approx(-10.0)

    def test_commission_and_slippage_reduce_pnl(self):
        pnl = compute_pnl_currency(100.0, 110.0, "long", 1.0, 2.0, 1.0)
        assert pnl == pytest.approx(7.0)  # 10 - 2 - 1

    def test_pnl_from_fill_engine(self):
        """End-to-end: FillEngine-computed P&L matches manual formula."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", qty=10.0, target=110.0)]
        engine.process_bar(_bar(100.0, 101.0, 99.0, 100.5, "2020-01-01"), orders)
        trades = engine.process_bar(_bar(105.0, 115.0, 104.0, 112.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].pnl_currency == pytest.approx(100.0)  # (110-100)*10


# ── R calculation ─────────────────────────────────────────────────────────────

class TestPnlR:
    def test_long_1r(self):
        """Entry 100, stop 90, exit 110, long → R = 1.0"""
        r = compute_pnl_r(100.0, 110.0, 90.0, "long", 1.0, 0.0, 0.0)
        assert r == pytest.approx(1.0)

    def test_long_half_r(self):
        r = compute_pnl_r(100.0, 105.0, 90.0, "long", 1.0, 0.0, 0.0)
        assert r == pytest.approx(0.5)

    def test_long_negative_r(self):
        r = compute_pnl_r(100.0, 92.0, 90.0, "long", 1.0, 0.0, 0.0)
        assert r == pytest.approx(-0.8)

    def test_short_1r(self):
        r = compute_pnl_r(100.0, 90.0, 110.0, "short", 1.0, 0.0, 0.0)
        assert r == pytest.approx(1.0)

    def test_none_when_no_stop(self):
        r = compute_pnl_r(100.0, 110.0, None, "long", 1.0, 0.0, 0.0)
        assert r is None

    def test_none_when_stop_above_entry_for_long(self):
        """Invalid stop placement → undefined R."""
        r = compute_pnl_r(100.0, 110.0, 105.0, "long", 1.0, 0.0, 0.0)
        assert r is None

    def test_r_from_fill_engine(self):
        """FillEngine-computed pnl_r matches manual formula (stop defined)."""
        engine = FillEngine(tick_size=0.01)
        orders = [_market("long", qty=1.0, stop=90.0, target=110.0)]
        engine.process_bar(_bar(100.0, 101.0, 99.0, 100.5, "2020-01-01"), orders)
        trades = engine.process_bar(_bar(105.0, 115.0, 104.0, 112.0, "2020-01-02"), [])
        assert trades[0].pnl_r == pytest.approx(1.0)   # (110-100)/(100-90) = 1.0


# ── close_all ─────────────────────────────────────────────────────────────────

class TestFlattenPosition:
    def test_flatten_closes_at_bar_close(self):
        """Long trade exits at bar close; pnl, status, and flags are correct."""
        bar = pd.Series(
            {"open": 100.0, "high": 110.0, "low": 98.0, "close": 105.0},
            name=pd.Timestamp("2024-01-05"),
        )
        trade = Trade(
            entry_price=100.0, side="long", quantity=10,
            stop_price=90.0, target_price=120.0,
        )
        result = flatten_position(trade, bar)
        assert result.exit_price     == pytest.approx(105.0)
        assert result.exit_reason    == "manual"
        assert result.ambiguity_flag == 0
        assert result.pnl_currency   == pytest.approx(50.0)   # (105 - 100) * 10
        assert result.status         == "closed"

    def test_flatten_short_position(self):
        """Short trade exits at bar close; pnl is positive when price fell."""
        bar = pd.Series(
            {"open": 100.0, "high": 102.0, "low": 96.0, "close": 98.0},
            name=pd.Timestamp("2024-01-05"),
        )
        trade = Trade(
            entry_price=100.0, side="short", quantity=10,
            stop_price=110.0, target_price=85.0,
        )
        result = flatten_position(trade, bar)
        assert result.exit_price   == pytest.approx(98.0)
        assert result.pnl_currency == pytest.approx(20.0)   # (100 - 98) * 10
        assert result.exit_reason  == "manual"
        assert result.status       == "closed"


class TestCloseAll:
    def test_closes_all_positions_at_close(self):
        engine = FillEngine(tick_size=0.01)
        o1 = _market("long",  qty=1.0)
        o2 = _market("short", qty=2.0)
        engine.process_bar(_bar(100.0, 101.0, 99.0, 100.5, "2020-01-01"), [o1, o2])
        assert len(engine.open_positions) == 2

        bar    = _bar(105.0, 108.0, 104.0, 106.0, "2020-01-02")
        trades = engine.close_all(bar)
        assert len(trades) == 2
        assert all(t.exit_reason == "manual" for t in trades)
        assert all(t.exit_price == pytest.approx(106.0) for t in trades)
        assert len(engine.open_positions) == 0
