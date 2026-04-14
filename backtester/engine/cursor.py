"""BarCursor — controls which bars are currently visible to the UI."""

from __future__ import annotations

import pandas as pd


class BarCursor:
    """Advances through a bar DataFrame one bar at a time.

    Parameters
    ----------
    bars:
        Full OHLCV DataFrame.  Never sliced externally; the cursor exposes
        only the visible prefix via :attr:`visible_bars`.
    """

    def __init__(self, bars: pd.DataFrame) -> None:
        if bars.empty:
            raise ValueError("BarCursor requires a non-empty DataFrame.")
        self._bars = bars
        self._index: int = 0
        self._in_review_mode: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def bars(self) -> pd.DataFrame:
        """Full bar data — never sliced."""
        return self._bars

    @property
    def current_index(self) -> int:
        """Current 0-based bar index."""
        return self._index

    @property
    def visible_bars(self) -> pd.DataFrame:
        """Bars up to and including *current_index* (no future data)."""
        return self._bars.iloc[: self._index + 1]

    @property
    def is_complete(self) -> bool:
        """True when the cursor is at the last bar."""
        return self._index == len(self._bars) - 1

    def advance(self) -> None:
        """Move forward by one bar.

        Raises
        ------
        StopIteration
            When already at the last bar.
        """
        if self.is_complete:
            raise StopIteration("No more bars to advance.")
        self._index += 1

    def reset(self) -> None:
        """Return to the first bar."""
        self._index = 0

    # ------------------------------------------------------------------
    # Review mode
    # ------------------------------------------------------------------

    @property
    def in_review_mode(self) -> bool:
        """True when review mode is active (step_back is permitted)."""
        return self._in_review_mode

    def enter_review_mode(self) -> None:
        """Enable review mode, allowing :meth:`step_back`.

        Raises
        ------
        RuntimeError
            If fewer than one bar has been advanced (nothing to step back to).
        """
        if self._index < 1:
            raise RuntimeError(
                "Cannot enter review mode before advancing at least one bar."
            )
        self._in_review_mode = True

    def exit_review_mode(self) -> None:
        """Disable review mode."""
        self._in_review_mode = False

    def step_back(self) -> None:
        """Decrement current_index by one bar.

        Raises
        ------
        RuntimeError
            If review mode is not active, or already at bar 0.
        """
        if not self._in_review_mode:
            raise RuntimeError("step_back not allowed outside review mode")
        if self._index == 0:
            raise RuntimeError("Already at the first bar.")
        self._index -= 1
