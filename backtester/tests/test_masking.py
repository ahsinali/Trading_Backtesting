"""Tests for mask_series."""

from __future__ import annotations

import pytest

from backtester.engine.masking import mask_series


def test_mask_at_various_indices(synthetic_bars):
    series = synthetic_bars["close"]

    for idx in (0, 4, 9):
        result = mask_series(series, idx)
        assert len(result) == idx + 1
        # Values match the original series prefix
        assert list(result) == list(series.iloc[: idx + 1])


def test_masking_invariant_holds(synthetic_bars):
    """mask_series raises AssertionError when the invariant would be violated.

    We test the happy path (invariant holds) for every bar index to confirm
    the assertion never fires on valid inputs.
    """
    series = synthetic_bars["close"]
    for idx in range(len(series)):
        result = mask_series(series, idx)
        assert len(result) == idx + 1


def test_masking_invariant_violation():
    """Verify the assertion fires when the series is shorter than current_index."""
    import pandas as pd

    short = pd.Series([1.0, 2.0, 3.0])
    # Requesting index 5 on a 3-element series must violate the invariant
    with pytest.raises(AssertionError, match="Masking invariant violated"):
        mask_series(short, 5)
