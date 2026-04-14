"""Shared pytest fixtures for the backtester test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    """A 10-row OHLCV DataFrame with a DatetimeIndex.

    All price constraints are satisfied:
      high >= low, high >= open, high >= close,
      low <= open, low <= close.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = {
        # open, high, low, close chosen so every row is valid
        "open":  [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
        "high":  [105, 107, 106, 108, 110, 109, 111, 113, 112, 114],
        "low":   [ 99, 101, 100, 102, 104, 103, 105, 107, 106, 108],
        "close": [102, 104, 103, 105, 107, 106, 108, 110, 109, 111],
        "volume":[1000, 1100, 900, 1200, 1300, 800, 1400, 1500, 950, 1600],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "datetime"
    return df
