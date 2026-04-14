"""AmbiguityEngine — resolves bars that hit both stop and target simultaneously."""

from __future__ import annotations

from typing import Literal

import pandas as pd

AmbiguityPolicy = Literal["default", "conservative", "optimistic", "unknown"]


class AmbiguityEngine:
    """Apply a configurable ambiguity policy when both stop and target are hit.

    Parameters
    ----------
    policy:
        ``'default'``     — time-priority path (OHLC order).
        ``'conservative'``— always fill stop (worst case for the trader).
        ``'optimistic'``  — always fill target (best case for the trader).
        ``'unknown'``     — defer exit to the next unambiguous bar.
    """

    def __init__(self, policy: AmbiguityPolicy = "default") -> None:
        self.policy = policy

    def resolve(
        self,
        bar:          pd.Series,
        stop_price:   float,
        target_price: float,
        side:         str,
    ) -> tuple[str, bool]:
        """Determine how to close a position when both legs are hit on *bar*.

        Parameters
        ----------
        bar:
            The OHLCV bar (must contain ``'open'`` and ``'close'`` keys).
        stop_price:
            OCO stop level.
        target_price:
            OCO target level.
        side:
            ``'long'`` or ``'short'``.

        Returns
        -------
        tuple[str, bool]
            ``(reason, ambiguity_flag)`` where *reason* is one of
            ``'stop'``, ``'target'``, or ``'ambiguous'``.
            ``'ambiguous'`` tells the FillEngine to defer the exit to the
            next bar without closing the position now.
            *ambiguity_flag* is ``True`` whenever the policy had to break a
            tie (i.e. the outcome was not deterministic from bar structure).
        """
        if self.policy == "conservative":
            return ("stop", True)

        if self.policy == "optimistic":
            return ("target", True)

        if self.policy == "unknown":
            return ("ambiguous", True)

        # default: time-priority O→H→L→C (bullish bar) or O→L→H→C (bearish bar)
        bullish = float(bar["close"]) >= float(bar["open"])
        if side == "long":
            # Long: target is above entry, stop is below.
            # Bullish bar reaches High before Low → target is hit first.
            reason = "target" if bullish else "stop"
        else:
            # Short: target is below entry, stop is above.
            # Bullish bar reaches High before Low → stop is hit first.
            reason = "stop" if bullish else "target"

        return (reason, False)
