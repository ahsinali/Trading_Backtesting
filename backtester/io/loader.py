"""CSV loader for OHLCV bar data."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


_DATETIME_CANDIDATES = ("datetime", "date", "time", "timestamp")
_REQUIRED_OHLC = ("open", "high", "low", "close")


def load_csv(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV file and return a validated DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Index is a DatetimeIndex (parsed from the datetime column).
        Columns include ``open``, ``high``, ``low``, ``close``; ``volume`` is
        optional.  ``df.attrs`` contains metadata keys:
        ``source_path``, ``row_count``, ``date_range``, ``checksum``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the CSV is missing required columns, contains null values in OHLC,
        or violates any OHLCV price-integrity constraint.
    """
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw_bytes = raw_path.read_bytes()
    checksum = hashlib.sha256(raw_bytes).hexdigest()

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- locate and parse datetime column ---
    dt_col = None
    for candidate in _DATETIME_CANDIDATES:
        if candidate in df.columns:
            dt_col = candidate
            break
    if dt_col is None:
        raise ValueError(
            f"CSV must contain a datetime column named one of: "
            f"{_DATETIME_CANDIDATES}.  Found columns: {list(df.columns)}"
        )

    try:
        df[dt_col] = pd.to_datetime(df[dt_col])
    except Exception as exc:
        raise ValueError(f"Could not parse '{dt_col}' column as datetime: {exc}") from exc

    df = df.set_index(dt_col)
    df.index.name = "datetime"

    # --- check required columns ---
    missing = [c for c in _REQUIRED_OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required OHLC column(s): {missing}")

    # --- null check ---
    ohlc = df[list(_REQUIRED_OHLC)]
    null_counts = ohlc.isnull().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"OHLC columns contain null values: {bad}")

    # --- price integrity ---
    _check_price_integrity(df)

    # --- attach metadata ---
    df.attrs["source_path"] = str(raw_path.resolve())
    df.attrs["row_count"] = len(df)
    df.attrs["date_range"] = (df.index.min(), df.index.max())
    df.attrs["checksum"] = checksum

    return df


def _check_price_integrity(df: pd.DataFrame) -> None:
    """Raise ValueError if any OHLC price relationship is violated."""
    checks = {
        "high >= low": df["high"] >= df["low"],
        "high >= open": df["high"] >= df["open"],
        "high >= close": df["high"] >= df["close"],
        "low <= open": df["low"] <= df["open"],
        "low <= close": df["low"] <= df["close"],
    }
    for rule, mask in checks.items():
        bad_rows = df.index[~mask].tolist()
        if bad_rows:
            raise ValueError(
                f"Price integrity violation ({rule}) at "
                f"{len(bad_rows)} row(s), first offender: {bad_rows[0]}"
            )
