"""Session manifest — records all parameters needed to reproduce a session."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionManifest:
    """Immutable record of a backtesting session's configuration.

    Every field that influences reproducibility is stored here:
    data provenance, indicator settings, and (optionally) the
    anonymization parameters.

    Parameters
    ----------
    session_id:
        UUID string that uniquely identifies this session.
    symbol:
        Instrument name (e.g. ``"AAPL"``).
    timeframe:
        Bar timeframe string (e.g. ``"1D"``, ``"5m"``).
    bar_range:
        2-tuple of ISO date strings ``(first_bar_date, last_bar_date)``.
    data_checksum:
        SHA-256 hex digest of the raw CSV bytes.
    indicator_config:
        Dict of indicator parameters used (e.g. ``{"bb_period": 20, "sma_period": 20}``).
    anonymization_config:
        Dict returned by :meth:`Anonymizer.to_dict`, or ``None`` if the
        session was not anonymized.
    created_at:
        ISO-8601 timestamp when the session was created.
    config_hash:
        SHA-256 of the manifest's core fields (excluding this field itself,
        ``trade_count``, ``final_equity``, and ``status``).  Computed via
        :meth:`compute_config_hash` after construction.
    trade_count:
        Number of completed trades at session end (0 while active).
    final_equity:
        Cumulative P&L at session end (0.0 while active).
    status:
        ``'active'`` while the session is in progress; ``'complete'`` after
        :meth:`update_completion` is called.
    """

    session_id:           str
    symbol:               str
    timeframe:            str
    bar_range:            tuple[str, str]
    data_checksum:        str
    indicator_config:     dict[str, Any]
    anonymization_config: dict[str, Any] | None
    created_at:           str

    # Fields populated after construction / session end
    config_hash:  str   = field(default="")
    trade_count:  int   = field(default=0)
    final_equity: float = field(default=0.0)
    status:       str   = field(default="active")   # 'active' | 'complete'

    # ------------------------------------------------------------------
    # Config-hash helpers
    # ------------------------------------------------------------------

    def compute_config_hash(self) -> str:
        """Return the SHA-256 of the session's immutable configuration.

        Only the core provenance fields are hashed (session_id, symbol,
        timeframe, bar_range, data_checksum, indicator_config,
        anonymization_config, created_at).  Mutable completion fields
        (config_hash, trade_count, final_equity, status) are excluded so
        the hash remains stable across session updates.
        """
        payload = {
            "session_id":           self.session_id,
            "symbol":               self.symbol,
            "timeframe":            self.timeframe,
            "bar_range":            list(self.bar_range),
            "data_checksum":        self.data_checksum,
            "indicator_config":     self.indicator_config,
            "anonymization_config": self.anonymization_config,
            "created_at":           self.created_at,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def update_completion(self, trade_count: int, final_equity: float) -> None:
        """Mark the session complete and record final statistics.

        Also recomputes and updates :attr:`config_hash`.
        """
        self.trade_count  = trade_count
        self.final_equity = final_equity
        self.status       = "complete"
        self.config_hash  = self.compute_config_hash()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to a pretty-printed JSON string."""
        payload = {
            "session_id":           self.session_id,
            "symbol":               self.symbol,
            "timeframe":            self.timeframe,
            "bar_range":            list(self.bar_range),
            "data_checksum":        self.data_checksum,
            "indicator_config":     self.indicator_config,
            "anonymization_config": self.anonymization_config,
            "created_at":           self.created_at,
            "config_hash":          self.config_hash,
            "trade_count":          self.trade_count,
            "final_equity":         self.final_equity,
            "status":               self.status,
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SessionManifest":
        """Deserialise from a JSON string produced by :meth:`to_json`."""
        data = json.loads(json_str)
        data["bar_range"] = tuple(data["bar_range"])
        return cls(**data)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the manifest to a ``.json`` file at *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "SessionManifest":
        """Load a manifest from a ``.json`` file written by :meth:`save`."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())
