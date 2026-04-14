"""Phase 1 Definition-of-Done assertions.

These tests are the canonical acceptance criteria for Phase 1:
  - No future candles ever visible (masking invariant)
  - Anonymization is deterministic (same seed → same output)
  - 50k-bar advance performance within budget
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.engine.anonymization import Anonymizer
from backtester.engine.cursor import BarCursor
from backtester.engine.masking import mask_series
from backtester.io.loader import load_csv

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden.csv"


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_bars(n: int) -> pd.DataFrame:
    """Return a synthetic n-row OHLCV DataFrame."""
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


# ── DoD 1: No future candles visible ─────────────────────────────────────────

def test_no_future_candles_visible() -> None:
    """visible_bars length == current_index + 1 at every advance step.

    This is the fundamental masking invariant: after N advances, the cursor
    must expose exactly N+1 bars (bars 0..N inclusive) — never more.
    """
    bars   = load_csv(str(GOLDEN_PATH))
    cursor = BarCursor(bars)

    for _ in range(50):
        assert len(cursor.visible_bars) == cursor.current_index + 1, (
            f"Masking violation at index {cursor.current_index}: "
            f"{len(cursor.visible_bars)} visible bars, expected {cursor.current_index + 1}"
        )
        cursor.advance()

    # Check the state after the final advance too
    assert len(cursor.visible_bars) == cursor.current_index + 1


# ── DoD 2: Anonymization determinism ─────────────────────────────────────────

def test_anonymization_determinism() -> None:
    """Identical seed + config produces identical transform parameters.

    Two independent Anonymizer instances constructed with the same seed must
    produce the same (a, b) price coefficients and the same time_delta — so
    any session can be identically reproduced from its manifest.
    """
    bars = load_csv(str(GOLDEN_PATH))

    a1 = Anonymizer(seed=99)
    a1.anonymize_prices(bars)
    a1.anonymize_dates(bars)

    a2 = Anonymizer(seed=99)
    a2.anonymize_prices(bars)
    a2.anonymize_dates(bars)

    assert a1.a == a2.a, f"Scale factor mismatch: {a1.a} vs {a2.a}"
    assert a1.b == a2.b, f"Offset mismatch: {a1.b} vs {a2.b}"
    assert a1._time_delta == a2._time_delta, (
        f"Time delta mismatch: {a1._time_delta} vs {a2._time_delta}"
    )

    # Also verify that different seeds produce different results
    a3 = Anonymizer(seed=42)
    a3.anonymize_prices(bars)
    assert a1.a != a3.a, "Different seeds should produce different scale factors"


# ── DoD 3: 50k-bar advance performance ───────────────────────────────────────

def test_50k_bar_advance_performance() -> None:
    """100 advance + mask cycles on a 50 000-bar series must average < 100 ms.

    This measures the data-preparation work that runs on every bar advance:
    advancing the cursor and masking indicator arrays.  Qt painting is not
    included (it cannot run headlessly), but the data path must be fast enough
    that the full render stays well inside the 100 ms budget.
    """
    n      = 50_000
    bars   = _make_bars(n)
    cursor = BarCursor(bars)

    # Pre-compute a representative indicator series
    from backtester.engine.indicators import sma
    sma_series = sma(bars["close"], 20)

    # Advance to midpoint so the window is representative
    for _ in range(n // 2):
        cursor.advance()

    cycles  = 100
    t_start = time.perf_counter()
    for _ in range(cycles):
        if not cursor.is_complete:
            cursor.advance()
        idx    = cursor.current_index
        _masked = mask_series(sma_series, idx).to_numpy(dtype=np.float64)
    elapsed = time.perf_counter() - t_start

    avg_ms = elapsed / cycles * 1_000
    assert avg_ms < 100, (
        f"Average advance+mask time {avg_ms:.2f} ms exceeds 100 ms budget"
    )
