"""Tests for BarCursor."""

from __future__ import annotations

import pytest

from backtester.engine.cursor import BarCursor


def test_advance_increments_index(synthetic_bars):
    cursor = BarCursor(synthetic_bars)
    assert cursor.current_index == 0
    cursor.advance()
    assert cursor.current_index == 1


def test_stopiteration_at_end(synthetic_bars):
    cursor = BarCursor(synthetic_bars)
    # Advance to the last bar (index 9 for a 10-row frame)
    for _ in range(len(synthetic_bars) - 1):
        cursor.advance()
    assert cursor.is_complete
    with pytest.raises(StopIteration):
        cursor.advance()


def test_visible_bars_length(synthetic_bars):
    cursor = BarCursor(synthetic_bars)
    # Check invariant at every step from index 0 through 9
    for expected_index in range(len(synthetic_bars)):
        assert len(cursor.visible_bars) == cursor.current_index + 1
        assert len(cursor.visible_bars) == expected_index + 1
        if not cursor.is_complete:
            cursor.advance()


def test_reset(synthetic_bars):
    cursor = BarCursor(synthetic_bars)
    cursor.advance()
    cursor.advance()
    cursor.advance()
    assert cursor.current_index == 3
    cursor.reset()
    assert cursor.current_index == 0
    assert len(cursor.visible_bars) == 1


# ── Review Mode ───────────────────────────────────────────────────────────────

def test_review_mode_requires_at_least_one_advance(synthetic_bars):
    """enter_review_mode raises at index 0 — nothing to step back to."""
    cursor = BarCursor(synthetic_bars)
    assert cursor.current_index == 0
    with pytest.raises(RuntimeError):
        cursor.enter_review_mode()


def test_review_mode_available_mid_session(synthetic_bars):
    """Review mode can be entered at any point after the first advance."""
    cursor = BarCursor(synthetic_bars)

    # Advance a few bars — must not be complete
    cursor.advance()
    cursor.advance()
    cursor.advance()
    assert not cursor.is_complete

    # Enter review mode mid-session
    cursor.enter_review_mode()
    assert cursor.in_review_mode

    # Step back works
    idx_before = cursor.current_index
    cursor.step_back()
    assert cursor.current_index == idx_before - 1

    # Exit and resume
    cursor.exit_review_mode()
    assert not cursor.in_review_mode

    # Can advance again after exiting
    cursor.advance()
    assert cursor.current_index == idx_before


def test_step_back_cannot_go_before_zero(synthetic_bars):
    """step_back raises RuntimeError when already at bar 0."""
    cursor = BarCursor(synthetic_bars)
    cursor.advance()
    cursor.enter_review_mode()
    cursor.step_back()
    assert cursor.current_index == 0

    with pytest.raises(RuntimeError):
        cursor.step_back()


def test_step_back_blocked_outside_review(synthetic_bars):
    """step_back raises if review mode is not active."""
    cursor = BarCursor(synthetic_bars)
    cursor.advance()
    cursor.advance()
    with pytest.raises(RuntimeError):
        cursor.step_back()


def test_review_mode_at_session_end(synthetic_bars):
    """Review mode also works once all bars are exhausted."""
    cursor = BarCursor(synthetic_bars)
    for _ in range(len(synthetic_bars) - 1):
        cursor.advance()
    assert cursor.is_complete

    cursor.enter_review_mode()
    assert cursor.in_review_mode
    cursor.step_back()
    assert not cursor.is_complete
