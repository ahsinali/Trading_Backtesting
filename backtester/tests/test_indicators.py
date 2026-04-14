"""Golden-series tests for backtester/engine/indicators.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine.indicators import atr, ema, macd, rsi, sma


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def close_50() -> pd.Series:
    """50-bar synthetic close series built from a seeded random walk."""
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.standard_normal(50))
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlc_50(close_50) -> pd.DataFrame:
    """Synthetic OHLCV frame derived from close_50 for ATR tests."""
    rng = np.random.default_rng(7)
    noise = rng.uniform(0.1, 0.5, size=len(close_50))
    high = close_50 + noise
    low = close_50 - noise
    # Clamp so high >= low always holds
    df = pd.DataFrame(
        {
            "open": close_50,
            "high": high,
            "low": low,
            "close": close_50,
        }
    )
    return df


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------

def test_sma_golden_value(close_50):
    """SMA(5) at index 10 must match a manual window mean."""
    result = sma(close_50, period=5)
    expected = close_50.iloc[6:11].mean()  # window ending at index 10
    assert abs(result.iloc[10] - expected) < 1e-6, (
        f"SMA(5)[10] = {result.iloc[10]:.8f}, expected {expected:.8f}"
    )


def test_sma_leading_nans(close_50):
    """SMA(20) must have exactly 19 leading NaN values."""
    result = sma(close_50, period=20)
    nan_count = result.isna().sum()
    assert nan_count == 19, f"Expected 19 leading NaNs, got {nan_count}"
    assert not result.iloc[19:].isna().any(), "SMA(20) has NaN beyond warm-up"


def test_sma_length(close_50):
    result = sma(close_50, period=5)
    assert len(result) == len(close_50)


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

def test_ema_leading_nans(close_50):
    """EMA(10) must have exactly 9 leading NaN values."""
    result = ema(close_50, period=10)
    assert result.isna().sum() == 9


def test_ema_length(close_50):
    assert len(ema(close_50, period=5)) == len(close_50)


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

def test_rsi_bounded(close_50):
    """RSI values (non-NaN) must lie in [0, 100]."""
    result = rsi(close_50, period=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), (
        f"RSI out of bounds: min={valid.min():.4f}, max={valid.max():.4f}"
    )


def test_rsi_leading_nans(close_50):
    """RSI(14) must have exactly 14 leading NaN values."""
    result = rsi(close_50, period=14)
    assert result.isna().sum() == 14


def test_rsi_length(close_50):
    assert len(rsi(close_50)) == len(close_50)


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

def test_atr_non_negative(ohlc_50):
    """ATR values (non-NaN) must be >= 0."""
    result = atr(ohlc_50["high"], ohlc_50["low"], ohlc_50["close"], period=14)
    valid = result.dropna()
    assert (valid >= 0).all(), f"ATR has negative values: {valid[valid < 0]}"


def test_atr_leading_nans(ohlc_50):
    """ATR(14) must have exactly 14 leading NaN values."""
    result = atr(ohlc_50["high"], ohlc_50["low"], ohlc_50["close"], period=14)
    assert result.isna().sum() == 14


def test_atr_length(ohlc_50):
    result = atr(ohlc_50["high"], ohlc_50["low"], ohlc_50["close"])
    assert len(result) == len(ohlc_50)


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------

def test_macd_returns_three_series(close_50):
    result = macd(close_50)
    assert len(result) == 3
    macd_line, signal_line, hist = result
    assert len(macd_line) == len(close_50)
    assert len(signal_line) == len(close_50)
    assert len(hist) == len(close_50)


def test_macd_histogram_is_difference(close_50):
    """Histogram must equal macd_line - signal_line (where both are non-NaN)."""
    macd_line, signal_line, hist = macd(close_50)
    valid = ~macd_line.isna() & ~signal_line.isna()
    diff = (macd_line[valid] - signal_line[valid] - hist[valid]).abs()
    assert (diff < 1e-10).all()
