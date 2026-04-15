"""Golden-series tests for backtester/engine/indicators.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine.indicators import (
    IndicatorConfig,
    atr,
    bollinger_bands,
    compute_indicators,
    ema,
    keltner_channel,
    macd,
    rsi,
    sma,
)


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


# ---------------------------------------------------------------------------
# Keltner Channel tests
# ---------------------------------------------------------------------------

@pytest.fixture
def golden_bars(ohlc_50) -> pd.DataFrame:
    """ohlc_50 with a DatetimeIndex so it resembles real OHLCV data."""
    df = ohlc_50.copy()
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="1h")
    return df


def test_keltner_channel_shape(golden_bars):
    """keltner_channel() returns three series each with the same length as input."""
    upper, middle, lower = keltner_channel(
        golden_bars["high"], golden_bars["low"], golden_bars["close"]
    )
    assert len(upper) == len(golden_bars)
    assert len(middle) == len(golden_bars)
    assert len(lower) == len(golden_bars)


def test_keltner_upper_above_lower(golden_bars):
    """Upper band must always be >= lower band (where both are non-NaN)."""
    upper, middle, lower = keltner_channel(
        golden_bars["high"], golden_bars["low"], golden_bars["close"]
    )
    valid = ~upper.isna() & ~lower.isna()
    assert (upper[valid] >= lower[valid]).all()


def test_keltner_middle_between_bands(golden_bars):
    """Middle band must lie between upper and lower (where all are non-NaN)."""
    upper, middle, lower = keltner_channel(
        golden_bars["high"], golden_bars["low"], golden_bars["close"]
    )
    valid = ~upper.isna() & ~middle.isna() & ~lower.isna()
    assert (middle[valid] <= upper[valid]).all()
    assert (middle[valid] >= lower[valid]).all()


def test_compute_indicators_sma_mode(golden_bars):
    """compute_indicators in SMA mode returns sma1 and sma2 of correct length."""
    cfg  = IndicatorConfig(mode="sma", sma_period=10, sma_period_2=20)
    data = compute_indicators(golden_bars, cfg)
    assert set(data.keys()) == {"sma1", "sma2"}
    assert len(data["sma1"]) == len(golden_bars)
    assert len(data["sma2"]) == len(golden_bars)
    # sma1 should have 9 leading NaNs, sma2 should have 19
    assert data["sma1"].isna().sum() == 9
    assert data["sma2"].isna().sum() == 19


def test_compute_indicators_keltner_mode(golden_bars):
    """compute_indicators in Keltner mode returns kc_upper/middle/lower of correct length."""
    cfg  = IndicatorConfig(mode="keltner", keltner_ema_period=10, keltner_atr_period=5)
    data = compute_indicators(golden_bars, cfg)
    assert set(data.keys()) == {"kc_upper", "kc_middle", "kc_lower"}
    for key in ("kc_upper", "kc_middle", "kc_lower"):
        assert len(data[key]) == len(golden_bars)
    # All three should have the same NaN mask
    nan_mask_upper  = data["kc_upper"].isna()
    nan_mask_middle = data["kc_middle"].isna()
    nan_mask_lower  = data["kc_lower"].isna()
    assert (nan_mask_upper == nan_mask_middle).all()
    assert (nan_mask_upper == nan_mask_lower).all()
