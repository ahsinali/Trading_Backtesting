"""Performance test: data-preparation for 50 000-bar series.

Scope
-----
Qt painting cannot run headlessly, so this test times the *data-preparation*
portion of render_visible() — masking indicator arrays and applying the
anonymization transform — which is the pure-Python/NumPy work that scales
with bar count.  The assertion is: 100 consecutive advance+prepare cycles on
a 50 000-bar series must each complete in under 100 ms on average.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from backtester.engine.cursor import BarCursor
from backtester.engine.indicators import bollinger_bands, rsi, sma
from backtester.engine.masking import mask_series


def _make_bars(n: int) -> pd.DataFrame:
    rng    = np.random.default_rng(0)
    closes = 100.0 + np.cumsum(rng.standard_normal(n))
    opens  = np.concatenate([[closes[0]], closes[:-1]])
    noise  = rng.uniform(0.1, 0.5, n)
    dates  = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        {
            "open":   opens,
            "high":   np.maximum(opens, closes) + noise,
            "low":    np.minimum(opens, closes) - noise,
            "close":  closes,
            "volume": np.ones(n) * 1_000,
        },
        index=dates,
    )


def _prepare_frame(
    cursor: BarCursor,
    indicator_series: dict[str, pd.Series],
) -> None:
    """Replicate the array-preparation work from ChartWidget.render_visible()."""
    idx     = cursor.current_index
    visible = cursor.visible_bars
    n       = len(visible)
    x       = np.arange(n, dtype=np.float64)

    visible["open"].to_numpy(dtype=np.float64)
    visible["high"].to_numpy(dtype=np.float64)
    visible["low"].to_numpy(dtype=np.float64)
    visible["close"].to_numpy(dtype=np.float64)

    for name, series in indicator_series.items():
        masked = mask_series(series, idx).to_numpy(dtype=np.float64)
        _ = ~np.isnan(masked)


def test_render_100_frames_avg_under_100ms() -> None:
    """100 advance+prepare cycles on 50 000 bars must average < 100 ms each."""
    n    = 50_000
    bars = _make_bars(n)

    close  = bars["close"]
    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
    indicator_series = {
        "BB_upper":  bb_u,
        "BB_middle": bb_m,
        "BB_lower":  bb_l,
        "SMA20":     sma(close, 20),
        "RSI14":     rsi(close, 14),
    }

    cursor = BarCursor(bars)
    # Advance to the midpoint so there is a full visible window
    for _ in range(n // 2):
        cursor.advance()

    cycles  = 100
    t_start = time.perf_counter()
    for _ in range(cycles):
        if not cursor.is_complete:
            cursor.advance()
        _prepare_frame(cursor, indicator_series)
    elapsed = time.perf_counter() - t_start

    avg_ms = elapsed / cycles * 1_000
    assert avg_ms < 100, (
        f"Average frame-preparation time {avg_ms:.2f} ms exceeds 100 ms limit"
    )
