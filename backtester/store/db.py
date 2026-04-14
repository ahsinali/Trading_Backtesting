"""SQLite trade store — persists completed trades for a backtesting session."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from backtester.sim.orders import Trade


_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    entry_datetime  TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_datetime   TEXT    NOT NULL,
    exit_price      REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    side            TEXT    NOT NULL,
    stop_price      REAL,
    target_price    REAL,
    exit_reason     TEXT,
    pnl_currency    REAL    NOT NULL,
    pnl_r           REAL,
    commission      REAL    DEFAULT 0,
    slippage        REAL    DEFAULT 0,
    tick_size       REAL    NOT NULL,
    execution_rule  TEXT    NOT NULL,
    ambiguity_flag  INTEGER NOT NULL,
    notes           TEXT,
    config_hash     TEXT    NOT NULL
);
"""

_INSERT_TRADE = """
INSERT INTO trades (
    session_id, symbol, timeframe,
    entry_datetime, entry_price,
    exit_datetime,  exit_price,
    quantity, side,
    stop_price, target_price, exit_reason,
    pnl_currency, pnl_r,
    commission, slippage, tick_size,
    execution_rule, ambiguity_flag,
    notes, config_hash
) VALUES (
    :session_id, :symbol, :timeframe,
    :entry_datetime, :entry_price,
    :exit_datetime,  :exit_price,
    :quantity, :side,
    :stop_price, :target_price, :exit_reason,
    :pnl_currency, :pnl_r,
    :commission, :slippage, :tick_size,
    :execution_rule, :ambiguity_flag,
    :notes, :config_hash
);
"""


def _trade_to_row(trade: Trade) -> dict[str, Any]:
    return {
        "session_id":     trade.session_id,
        "symbol":         trade.symbol,
        "timeframe":      trade.timeframe,
        "entry_datetime": trade.entry_datetime,
        "entry_price":    trade.entry_price,
        "exit_datetime":  trade.exit_datetime,
        "exit_price":     trade.exit_price,
        "quantity":       trade.quantity,
        "side":           trade.side,
        "stop_price":     trade.stop_price,
        "target_price":   trade.target_price,
        "exit_reason":    trade.exit_reason,
        "pnl_currency":   trade.pnl_currency,
        "pnl_r":          trade.pnl_r,
        "commission":     trade.commission,
        "slippage":       trade.slippage,
        "tick_size":      trade.tick_size,
        "execution_rule": trade.execution_rule,
        "ambiguity_flag": trade.ambiguity_flag,
        "notes":          trade.notes,
        "config_hash":    trade.config_hash,
    }


def _row_to_trade(row: sqlite3.Row) -> Trade:
    return Trade(
        session_id     = row["session_id"],
        symbol         = row["symbol"],
        timeframe      = row["timeframe"],
        entry_datetime = row["entry_datetime"],
        entry_price    = row["entry_price"],
        exit_datetime  = row["exit_datetime"],
        exit_price     = row["exit_price"],
        quantity       = row["quantity"],
        side           = row["side"],
        stop_price     = row["stop_price"],
        target_price   = row["target_price"],
        exit_reason    = row["exit_reason"],
        pnl_currency   = row["pnl_currency"],
        pnl_r          = row["pnl_r"],
        commission     = row["commission"],
        slippage       = row["slippage"],
        tick_size      = row["tick_size"],
        execution_rule = row["execution_rule"],
        ambiguity_flag = row["ambiguity_flag"],
        notes          = row["notes"],
        config_hash    = row["config_hash"],
        db_id          = row["id"],
    )


class TradeStore:
    """Thin SQLite wrapper for the ``trades`` table.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created (along with its parent
        directory) if it does not exist.
    """

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self.create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create the ``trades`` table if it does not already exist."""
        self._conn.execute(_CREATE_TRADES)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_trade(self, trade: Trade) -> None:
        """Insert *trade* into the database and set ``trade.db_id``."""
        cur = self._conn.execute(_INSERT_TRADE, _trade_to_row(trade))
        self._conn.commit()
        trade.db_id = cur.lastrowid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_trades_for_session(self, session_id: str) -> list[Trade]:
        """Return all trades for *session_id*, ordered by entry datetime."""
        cur = self._conn.execute(
            "SELECT * FROM trades WHERE session_id = ? ORDER BY entry_datetime, id",
            (session_id,),
        )
        return [_row_to_trade(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_equity_curve(self, session_id: str, output_path: str) -> None:
        """Write the equity curve CSV for *session_id* to *output_path*.

        Columns: ``trade_number``, ``cumulative_pnl``.
        """
        from backtester.store.logs import compute_equity_curve
        trades = self.get_trades_for_session(session_id)
        curve  = compute_equity_curve(trades)
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            fh.write("trade_number,cumulative_pnl\n")
            for i, v in curve.items():
                fh.write(f"{i},{v:.4f}\n")

    def export_csv(self, session_id: str, output_path: str) -> None:
        """Write all trades for *session_id* to a CSV file at *output_path*."""
        trades = self.get_trades_for_session(session_id)
        if not trades:
            return
        fieldnames = [
            "db_id", "session_id", "symbol", "timeframe",
            "entry_datetime", "entry_price", "exit_datetime", "exit_price",
            "quantity", "side", "stop_price", "target_price", "exit_reason",
            "pnl_currency", "pnl_r", "commission", "slippage", "tick_size",
            "execution_rule", "ambiguity_flag", "notes", "config_hash",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                writer.writerow({f: getattr(t, f) for f in fieldnames})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "TradeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
