"""Pure indicator functions — operate on full series, return full-length pd.Series.

Rules:
- No lookahead: all functions are causal.
- NaN fills cover warm-up periods.
- No side effects; all inputs are read-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average.

    Parameters
    ----------
    close:
        Full close-price series.
    period:
        Look-back window.

    Returns
    -------
    pd.Series
        Same length as *close*; first ``period - 1`` values are NaN.
    """
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (pandas ewm, adjust=False).

    Parameters
    ----------
    close:
        Full close-price series.
    period:
        Span parameter (α = 2 / (period + 1)).

    Returns
    -------
    pd.Series
        Same length as *close*; first ``period - 1`` values are NaN.
    """
    raw = close.ewm(span=period, adjust=False).mean()
    # Mask the warm-up period so behaviour matches other indicators
    result = raw.copy()
    result.iloc[: period - 1] = np.nan
    return result


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Parameters
    ----------
    close:
        Full close-price series.
    period:
        Look-back period (default 14).

    Returns
    -------
    pd.Series
        Values in [0, 100]; first ``period`` values are NaN.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Seed the first average with a simple mean over the initial window
    avg_gain = gain.copy().astype(float)
    avg_loss = loss.copy().astype(float)

    # Wilder smoothing implemented manually
    result = pd.Series(np.nan, index=close.index, dtype=float)

    # First valid average: simple mean of first `period` changes (indices 1..period)
    if len(close) <= period:
        return result

    seed_gain = gain.iloc[1 : period + 1].mean()
    seed_loss = loss.iloc[1 : period + 1].mean()

    avg_g = seed_gain
    avg_l = seed_loss

    # RSI at index `period` (the first fully-warmed-up bar)
    first_rsi_idx = period
    if avg_l == 0:
        result.iloc[first_rsi_idx] = 100.0
    else:
        rs = avg_g / avg_l
        result.iloc[first_rsi_idx] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for subsequent bars
    for i in range(period + 1, len(close)):
        avg_g = (avg_g * (period - 1) + gain.iloc[i]) / period
        avg_l = (avg_l * (period - 1) + loss.iloc[i]) / period
        if avg_l == 0:
            result.iloc[i] = 100.0
        else:
            rs = avg_g / avg_l
            result.iloc[i] = 100.0 - 100.0 / (1.0 + rs)

    # Clamp to [0, 100] (guards against floating-point drift)
    result = result.clip(lower=0.0, upper=100.0)
    return result


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder smoothing.

    True Range = max(high - low,
                     |high - prev_close|,
                     |low  - prev_close|)

    Parameters
    ----------
    high, low, close:
        Full OHLC series, all sharing the same index.
    period:
        Wilder smoothing period (default 14).

    Returns
    -------
    pd.Series
        Same length as inputs; first ``period`` values are NaN.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result = pd.Series(np.nan, index=close.index, dtype=float)

    if len(close) <= period:
        return result

    # Seed: simple mean of first `period` true ranges (index 1..period)
    seed_atr = tr.iloc[1 : period + 1].mean()
    result.iloc[period] = seed_atr

    avg = seed_atr
    for i in range(period + 1, len(close)):
        avg = (avg * (period - 1) + tr.iloc[i]) / period
        result.iloc[i] = avg

    return result


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence.

    Parameters
    ----------
    close:
        Full close-price series.
    fast, slow:
        EMA spans for the fast and slow lines.
    signal:
        EMA span for the signal line.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        ``(macd_line, signal_line, histogram)`` — all same length as *close*.
    """
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    # Mask warm-up (slow EMA needs slow-1 bars)
    macd_line.iloc[: slow - 1] = np.nan

    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Mask until signal line is also warmed up
    signal_line.iloc[: slow - 1 + signal - 1] = np.nan

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (upper, middle, lower).

    Parameters
    ----------
    close:
        Full close-price series.
    period:
        Rolling window for the middle band (default 20).
    std_dev:
        Number of standard deviations for the outer bands (default 2.0).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        ``(upper, middle, lower)`` — all same length as *close*.
        First ``period - 1`` values are NaN.
        Uses population std (ddof=0) to match TradingView's default.
    """
    middle = close.rolling(window=period, min_periods=period).mean()
    std    = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return upper, middle, lower
