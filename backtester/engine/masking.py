"""Indicator masking — ensures no future data leaks to the UI."""

from __future__ import annotations

import pandas as pd


def mask_series(series: pd.Series, current_index: int) -> pd.Series:
    """Return the visible prefix of *series* up to and including *current_index*.

    Parameters
    ----------
    series:
        A precomputed indicator series aligned to the full bar DataFrame.
    current_index:
        The cursor's current 0-based position.

    Returns
    -------
    pd.Series
        Slice ``series.iloc[:current_index + 1]``.

    Raises
    ------
    AssertionError
        If the returned slice does not have exactly ``current_index + 1``
        elements — signals a masking invariant violation.
    """
    result = series.iloc[: current_index + 1]
    assert len(result) == current_index + 1, (
        f"Masking invariant violated: expected {current_index + 1} elements, "
        f"got {len(result)}."
    )
    return result
