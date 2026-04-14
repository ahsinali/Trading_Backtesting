"""Phase 1 golden-series tests using the 200-bar synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backtester.engine.cursor import BarCursor
from backtester.engine.indicators import atr, rsi, sma
from backtester.engine.masking import mask_series
from backtester.io.loader import load_csv

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden.csv"


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def golden_cursor() -> BarCursor:
    """BarCursor advanced to bar 50 (index 50, 51 visible bars)."""
    bars = load_csv(str(GOLDEN_PATH))
    c = BarCursor(bars)
    for _ in range(50):
        c.advance()
    return c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_visible_bars_length(golden_cursor: BarCursor) -> None:
    """After 50 advances the cursor exposes exactly 51 visible bars."""
    assert len(golden_cursor.visible_bars) == 51


def test_sma20_no_nan_at_50(golden_cursor: BarCursor) -> None:
    """SMA(20) must be defined (not NaN) once index ≥ 19."""
    sma_vals = sma(golden_cursor.bars["close"], 20)
    assert not np.isnan(sma_vals.iloc[50])


def test_rsi14_in_range_at_50(golden_cursor: BarCursor) -> None:
    """RSI(14) must lie in [0, 100] at index 50."""
    rsi_vals = rsi(golden_cursor.bars["close"], 14)
    val = float(rsi_vals.iloc[50])
    assert 0.0 <= val <= 100.0


def test_atr14_non_negative_at_50(golden_cursor: BarCursor) -> None:
    """ATR(14) must be non-negative at index 50."""
    atr_vals = atr(
        golden_cursor.bars["high"],
        golden_cursor.bars["low"],
        golden_cursor.bars["close"],
        14,
    )
    assert float(atr_vals.iloc[50]) >= 0.0


def test_step_back_raises_outside_review() -> None:
    """step_back() must raise RuntimeError when not in review mode."""
    bars = load_csv(str(GOLDEN_PATH))
    c = BarCursor(bars)
    c.advance()
    with pytest.raises(RuntimeError, match="review mode"):
        c.step_back()


def test_step_back_works_in_review() -> None:
    """step_back() decrements index by 1 when review mode is active."""
    bars = load_csv(str(GOLDEN_PATH))
    c = BarCursor(bars)
    c.advance()
    c.advance()
    assert c.current_index == 2
    c.enter_review_mode()
    c.step_back()
    assert c.current_index == 1


def test_masking_invariant_across_advances() -> None:
    """mask_series length == current_index + 1 for every bar 0..29."""
    bars = load_csv(str(GOLDEN_PATH))
    c = BarCursor(bars)
    sma_full = sma(c.bars["close"], 20)
    for _ in range(30):
        masked = mask_series(sma_full, c.current_index)
        assert len(masked) == c.current_index + 1
        if not c.is_complete:
            c.advance()
