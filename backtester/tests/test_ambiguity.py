"""Phase 2B tests: AmbiguityEngine, limit/stop entry fills, logs."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.sim.ambiguity import AmbiguityEngine
from backtester.sim.fills import FillEngine
from backtester.sim.orders import Order, snap_to_tick
from backtester.sim.slippage import SLIPPAGE_TABLE, get_commission, get_slippage
from backtester.store.logs import compute_equity_curve, compute_summary_stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(open_: float, high: float, low: float, close: float,
         date: str = "2020-01-01") -> pd.Series:
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close},
        name=pd.Timestamp(date),
    )


def _market(side: str = "long", qty: float = 1.0,
            stop: float | None = None, target: float | None = None) -> Order:
    return Order.market(
        session_id="test", side=side, quantity=qty,
        tick_size=0.01, created_at_index=0,
        stop_price=stop, target_price=target,
    )


def _limit(side: str, price: float, qty: float = 1.0,
           stop: float | None = None, target: float | None = None) -> Order:
    return Order.limit(
        session_id="test", side=side, quantity=qty,
        limit_price=price, tick_size=0.01, created_at_index=0,
        stop_price=stop, target_price=target,
    )


def _stop_entry(side: str, trigger: float, qty: float = 1.0,
                bracket_stop: float | None = None,
                bracket_target: float | None = None) -> Order:
    return Order.stop_entry(
        session_id="test", side=side, quantity=qty,
        stop_price=trigger, tick_size=0.01, created_at_index=0,
        bracket_stop=bracket_stop, bracket_target=bracket_target,
    )


def _make_trade(pnl: float, pnl_r: float | None = None):
    """Minimal Trade-like object for logs tests."""
    from backtester.sim.orders import Trade
    return Trade(
        session_id="s", symbol="X", timeframe="1D",
        entry_datetime="2020-01-01", entry_price=100.0,
        exit_datetime="2020-01-02",  exit_price=110.0,
        quantity=1.0, side="long",
        stop_price=None, target_price=None,
        exit_reason="target",
        pnl_currency=pnl, pnl_r=pnl_r,
        commission=0.0, slippage=0.0, tick_size=0.01,
        execution_rule="default", ambiguity_flag=0,
        notes=None, config_hash="abc",
    )


# ── AmbiguityEngine — unit tests ──────────────────────────────────────────────

class TestAmbiguityEngine:
    # Scenario C: both hit, default, bullish bar
    def test_default_bullish_long_target(self):
        eng = AmbiguityEngine("default")
        bar = _bar(100.0, 120.0, 85.0, 115.0)  # close > open → bullish
        reason, flag = eng.resolve(bar, stop_price=90.0, target_price=115.0, side="long")
        assert reason == "target"
        assert flag is False

    def test_default_bearish_long_stop(self):
        eng = AmbiguityEngine("default")
        bar = _bar(108.0, 120.0, 85.0, 86.0)   # close < open → bearish
        reason, flag = eng.resolve(bar, stop_price=90.0, target_price=115.0, side="long")
        assert reason == "stop"
        assert flag is False

    def test_default_bullish_short_stop(self):
        eng = AmbiguityEngine("default")
        bar = _bar(100.0, 120.0, 80.0, 115.0)  # bullish
        reason, flag = eng.resolve(bar, stop_price=115.0, target_price=85.0, side="short")
        assert reason == "stop"
        assert flag is False

    def test_default_bearish_short_target(self):
        eng = AmbiguityEngine("default")
        bar = _bar(100.0, 120.0, 80.0, 82.0)   # bearish
        reason, flag = eng.resolve(bar, stop_price=115.0, target_price=85.0, side="short")
        assert reason == "target"
        assert flag is False

    # Scenario D: conservative
    def test_conservative_always_stop(self):
        eng = AmbiguityEngine("conservative")
        bar = _bar(100.0, 120.0, 85.0, 115.0)  # bullish — would be target in default
        reason, flag = eng.resolve(bar, stop_price=90.0, target_price=115.0, side="long")
        assert reason == "stop"
        assert flag is True

    def test_optimistic_always_target(self):
        eng = AmbiguityEngine("optimistic")
        bar = _bar(108.0, 120.0, 85.0, 86.0)   # bearish — would be stop in default
        reason, flag = eng.resolve(bar, stop_price=90.0, target_price=115.0, side="long")
        assert reason == "target"
        assert flag is True

    # Scenario E: unknown
    def test_unknown_returns_ambiguous(self):
        eng = AmbiguityEngine("unknown")
        bar = _bar(100.0, 120.0, 85.0, 115.0)
        reason, flag = eng.resolve(bar, stop_price=90.0, target_price=115.0, side="long")
        assert reason == "ambiguous"
        assert flag is True


# ── Limit entry fills (Scenario A/B: no ambiguity — only one bracket hit) ─────

class TestLimitFills:
    def test_long_limit_fills_when_touched(self):
        """Buy limit fills when bar low ≤ limit ≤ bar high."""
        engine = FillEngine(tick_size=0.01)
        orders = [_limit("long", 98.0)]
        bar = _bar(99.0, 101.0, 97.0, 100.0)   # low=97 ≤ 98 ≤ high=101
        engine.process_bar(bar, orders)
        assert orders[0].status == "filled"
        assert engine.open_positions[0].entry_price == pytest.approx(98.0)

    def test_long_limit_no_fill_above_high(self):
        """Buy limit above bar high — no fill."""
        engine = FillEngine(tick_size=0.01)
        orders = [_limit("long", 102.0)]
        engine.process_bar(_bar(99.0, 101.0, 97.0, 100.0), orders)
        assert orders[0].status == "pending"

    def test_short_limit_fills_when_touched(self):
        """Sell limit fills when bar low ≤ limit ≤ bar high."""
        engine = FillEngine(tick_size=0.01)
        orders = [_limit("short", 101.0)]
        bar = _bar(99.0, 103.0, 98.0, 100.0)
        engine.process_bar(bar, orders)
        assert orders[0].status == "filled"
        assert engine.open_positions[0].entry_price == pytest.approx(101.0)

    def test_limit_no_slippage(self):
        """Limit fills carry zero slippage cost."""
        engine = FillEngine(tick_size=0.01, slippage=0.05)
        orders = [_limit("long", 100.0)]
        engine.process_bar(_bar(100.0, 101.0, 99.0, 100.5), orders)
        assert engine.open_positions[0].slippage_pnl_cost == pytest.approx(0.0)

    def test_limit_pnl_no_slippage_deduction(self):
        """End-to-end: limit fill P&L is gross (no slippage)."""
        engine = FillEngine(tick_size=0.01, slippage=0.05)
        orders = [_limit("long", 100.0, target=110.0)]
        engine.process_bar(_bar(100.0, 101.0, 99.0, 100.5, "2020-01-01"), orders)
        trades = engine.process_bar(_bar(108.0, 115.0, 107.0, 112.0, "2020-01-02"), [])
        assert len(trades) == 1
        assert trades[0].pnl_currency == pytest.approx(10.0)  # (110-100)*1, no slip


# ── Stop-entry fills ──────────────────────────────────────────────────────────

class TestStopEntryFills:
    def test_long_stop_buy_fills_when_triggered(self):
        """Long stop-buy triggers when bar high ≥ trigger price."""
        engine = FillEngine(tick_size=0.01)
        orders = [_stop_entry("long", 105.0)]
        bar = _bar(100.0, 108.0, 99.0, 107.0)  # high=108 ≥ 105
        engine.process_bar(bar, orders)
        assert orders[0].status == "filled"
        assert engine.open_positions[0].entry_price == pytest.approx(105.0)

    def test_long_stop_buy_no_fill_below_trigger(self):
        """Long stop-buy not triggered when bar high < trigger."""
        engine = FillEngine(tick_size=0.01)
        orders = [_stop_entry("long", 110.0)]
        engine.process_bar(_bar(100.0, 108.0, 99.0, 107.0), orders)
        assert orders[0].status == "pending"

    def test_short_stop_sell_fills_when_triggered(self):
        """Short stop-sell triggers when bar low ≤ trigger price."""
        engine = FillEngine(tick_size=0.01)
        orders = [_stop_entry("short", 95.0)]
        bar = _bar(100.0, 101.0, 92.0, 94.0)   # low=92 ≤ 95
        engine.process_bar(bar, orders)
        assert orders[0].status == "filled"
        assert engine.open_positions[0].entry_price == pytest.approx(95.0)

    def test_stop_entry_embeds_slippage_in_price(self):
        """Stop-buy fill price = trigger + slippage (1 tick)."""
        engine = FillEngine(tick_size=0.01, slippage=0.01)
        orders = [_stop_entry("long", 105.0)]
        bar = _bar(100.0, 108.0, 99.0, 107.0)
        engine.process_bar(bar, orders)
        assert engine.open_positions[0].entry_price == pytest.approx(105.01)

    def test_short_stop_entry_embeds_slippage(self):
        """Short stop-sell fill price = trigger − slippage."""
        engine = FillEngine(tick_size=0.01, slippage=0.01)
        orders = [_stop_entry("short", 95.0)]
        engine.process_bar(_bar(100.0, 101.0, 92.0, 94.0), orders)
        assert engine.open_positions[0].entry_price == pytest.approx(94.99)


# ── set_execution_rule ────────────────────────────────────────────────────────

class TestSetExecutionRule:
    def test_can_change_rule_before_first_order(self):
        engine = FillEngine(tick_size=0.01, execution_rule="default")
        engine.set_execution_rule("conservative")
        assert engine._execution_rule == "conservative"
        # Should now behave conservatively (stop wins) on ambiguous bar
        orders = [_market("long", stop=90.0, target=115.0)]
        engine.process_bar(_bar(100.0, 101.0, 99.0, 100.5, "2020-01-01"), orders)
        bar = _bar(108.0, 120.0, 85.0, 115.0, "2020-01-02")  # bullish, both hit
        trades = engine.process_bar(bar, [])
        assert trades[0].exit_reason == "stop"


# ── Slippage helpers ──────────────────────────────────────────────────────────

class TestSlippage:
    def test_equity_slippage_one_tick(self):
        assert get_slippage("equity", 0.01) == pytest.approx(0.01)

    def test_futures_slippage_one_tick(self):
        assert get_slippage("futures", 0.25) == pytest.approx(0.25)

    def test_fx_slippage_two_ticks(self):
        assert get_slippage("fx", 0.0001) == pytest.approx(0.0002)

    def test_unknown_class_falls_back_to_equity(self):
        assert get_slippage("crypto", 1.0) == pytest.approx(1.0)

    def test_equity_commission(self):
        assert get_commission("equity", 100.0) == pytest.approx(0.50)

    def test_futures_commission(self):
        assert get_commission("futures", 3.0) == pytest.approx(7.50)

    def test_fx_commission_zero(self):
        assert get_commission("fx", 10.0) == pytest.approx(0.0)


# ── compute_equity_curve ──────────────────────────────────────────────────────

class TestEquityCurve:
    def test_empty_trades_returns_empty(self):
        curve = compute_equity_curve([])
        assert len(curve) == 0

    def test_single_trade(self):
        curve = compute_equity_curve([_make_trade(100.0)])
        assert len(curve) == 1
        assert curve.iloc[0] == pytest.approx(100.0)

    def test_cumsum_correct(self):
        trades = [_make_trade(100.0), _make_trade(-30.0), _make_trade(50.0)]
        curve  = compute_equity_curve(trades)
        assert curve.tolist() == pytest.approx([100.0, 70.0, 120.0])

    def test_index_is_one_based(self):
        curve = compute_equity_curve([_make_trade(1.0), _make_trade(2.0)])
        assert list(curve.index) == [1, 2]


# ── compute_summary_stats ─────────────────────────────────────────────────────

class TestSummaryStats:
    def test_empty_returns_zeros(self):
        stats = compute_summary_stats([])
        assert stats["net_pnl"] == pytest.approx(0.0)
        assert stats["trade_count"] == 0

    def test_net_pnl(self):
        trades = [_make_trade(100.0), _make_trade(-40.0)]
        stats  = compute_summary_stats(trades)
        assert stats["net_pnl"] == pytest.approx(60.0)

    def test_win_rate(self):
        trades = [_make_trade(10.0), _make_trade(10.0), _make_trade(-5.0)]
        stats  = compute_summary_stats(trades)
        assert stats["win_rate"] == pytest.approx(2 / 3)

    def test_profit_factor(self):
        trades = [_make_trade(60.0), _make_trade(-20.0)]
        stats  = compute_summary_stats(trades)
        assert stats["profit_factor"] == pytest.approx(3.0)

    def test_profit_factor_inf_no_losses(self):
        stats = compute_summary_stats([_make_trade(50.0)])
        assert stats["profit_factor"] == float("inf")

    def test_max_drawdown_negative(self):
        # Equity: 10, 5, 15 → drawdown at bar 2 = 5 - 10 = -5
        trades = [_make_trade(10.0), _make_trade(-5.0), _make_trade(10.0)]
        stats  = compute_summary_stats(trades)
        assert stats["max_drawdown"] == pytest.approx(-5.0)

    def test_expectancy_r(self):
        trades = [_make_trade(10.0, pnl_r=2.0), _make_trade(-5.0, pnl_r=-1.0)]
        stats  = compute_summary_stats(trades)
        assert stats["expectancy_r"] == pytest.approx(0.5)  # (2 + -1) / 2

    def test_trade_count(self):
        trades = [_make_trade(1.0)] * 7
        stats  = compute_summary_stats(trades)
        assert stats["trade_count"] == 7
