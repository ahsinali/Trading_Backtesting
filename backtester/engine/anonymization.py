"""Deterministic anonymization of price and date data."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd


class Anonymizer:
    """Applies a deterministic, seed-controlled affine price transform and
    constant date shift to a bar DataFrame, removing identifying information
    while preserving the shape and statistics of the series.

    Price transform
    ---------------
    ``P' = a * P + b``

    ``a`` is drawn from ``Uniform(0.5, 2.0)`` using the supplied *seed*,
    and ``b`` is chosen so that the close at *start_index* maps to exactly
    1 000.0.

    Date transform
    --------------
    Every timestamp is shifted by a constant ``timedelta`` so that the first
    bar's date lands on *anchor_date*.  When *preserve_seasonality* is
    ``True`` the same constant delta is used (bar spacing, weekday patterns,
    and within-year seasonality are all preserved); when ``False`` the
    timestamps are shifted by a randomised whole-number-of-days offset that
    still satisfies the first-bar anchor constraint (future extension).

    All computed parameters (``a``, ``b``, ``time_delta``) are stored as
    instance attributes so they can be recorded in the session manifest and
    used for restoration.
    """

    def __init__(
        self,
        seed: int = 42,
        anchor_date: str = "1990-01-01",
        preserve_seasonality: bool = True,
    ) -> None:
        self.seed                  = seed
        self.anchor_date           = anchor_date
        self.preserve_seasonality  = preserve_seasonality

        # Populated by anonymize_prices / anonymize_dates
        self.a: float | None               = None
        self.b: float | None               = None
        self._time_delta: datetime.timedelta | None = None

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------

    def anonymize_prices(
        self,
        df: pd.DataFrame,
        start_index: int = 0,
    ) -> pd.DataFrame:
        """Return a copy of *df* with OHLC columns rescaled.

        Picks a random scale factor *a* (seeded, reproducible) then solves
        for *b* so that ``close.iloc[start_index]`` maps to 1 000.0.

        Parameters
        ----------
        df:
            DataFrame with columns ``open``, ``high``, ``low``, ``close``.
        start_index:
            Row index of the anchor bar (whose close becomes 1 000.0).
        """
        rng          = np.random.default_rng(self.seed)
        anchor_close = float(df["close"].iloc[start_index])
        if anchor_close == 0.0:
            raise ValueError(
                f"Close at start_index={start_index} is zero; "
                "affine transform is undefined."
            )

        self.a = float(rng.uniform(0.5, 2.0))
        self.b = 1_000.0 - self.a * anchor_close

        result = df.copy()
        for col in ("open", "high", "low", "close"):
            if col in result.columns:
                result[col] = self.a * result[col] + self.b

        return result

    def restore_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Invert the price transform: ``P = (P' - b) / a``.

        Requires :meth:`anonymize_prices` to have been called first.
        """
        if self.a is None or self.b is None:
            raise RuntimeError(
                "anonymize_prices() must be called before restore_prices()."
            )
        result = df.copy()
        for col in ("open", "high", "low", "close"):
            if col in result.columns:
                result[col] = (result[col] - self.b) / self.a
        return result

    def transform_prices_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply ``P' = a * P + b`` element-wise (NumPy array)."""
        if self.a is None or self.b is None:
            raise RuntimeError("anonymize_prices() must be called first.")
        return self.a * arr + self.b

    def transform_price(self, price: float) -> float:
        """Apply ``P' = a * P + b`` to a scalar."""
        if self.a is None or self.b is None:
            raise RuntimeError("anonymize_prices() must be called first.")
        return self.a * price + self.b

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def anonymize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with a shifted DatetimeIndex.

        Computes ``delta = anchor_date - df.index[0]`` and adds it to every
        timestamp, so the first bar lands on *anchor_date* exactly.
        All relative bar spacings are preserved.
        """
        if df.index.empty:
            self._time_delta = datetime.timedelta(0)
            return df.copy()

        anchor     = pd.Timestamp(self.anchor_date)
        first_date = pd.Timestamp(df.index[0])
        self._time_delta = (anchor - first_date).to_pytimedelta()

        result = df.copy()
        result.index = df.index + self._time_delta
        result.index.name = df.index.name
        return result

    def get_anonymized_dates(self, dates: list) -> list:
        """Shift a list of ``datetime`` / ``Timestamp`` objects by *time_delta*.

        Parameters
        ----------
        dates:
            List of ``datetime.datetime`` or ``pd.Timestamp`` objects.

        Raises
        ------
        RuntimeError
            If :meth:`anonymize_dates` has not been called yet.
        """
        if self._time_delta is None:
            raise RuntimeError("anonymize_dates() must be called first.")
        return [d + self._time_delta for d in dates]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all anonymization parameters."""
        return {
            "seed":                 self.seed,
            "anchor_date":          self.anchor_date,
            "preserve_seasonality": self.preserve_seasonality,
            "a":                    self.a,
            "b":                    self.b,
            "time_delta_days": (
                self._time_delta.total_seconds() / 86_400.0
                if self._time_delta is not None
                else None
            ),
        }
