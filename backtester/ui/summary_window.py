"""Session summary dialog — 3-tab view of equity curve, stats, and trade log."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from backtester.store.logs import compute_equity_curve, compute_summary_stats

# ── style constants ─────────────────────────────────────────────────────────
_BG      = "#131722"
_FG      = "#c8cdd4"
_HDR_BG  = "#252932"
_DIM     = "#555d6b"
_GREEN   = "#26a69a"
_RED     = "#ef5350"

_TABLE_CSS = f"""
    QTableWidget {{
        background: {_BG};
        color: {_FG};
        gridline-color: #2a2e39;
        border: none;
        font-size: 12px;
    }}
    QHeaderView::section {{
        background: {_HDR_BG};
        color: {_FG};
        border: none;
        padding: 4px 6px;
        font-size: 11px;
    }}
    QTableWidget::item:selected {{
        background: #1e3a5f;
    }}
"""

_DISPLAY_NAMES: dict[str, str] = {
    "net_pnl":            "Net P&L",
    "avg_trade_pnl":      "Avg Trade P&L",
    "win_rate":           "Win Rate",
    "profit_factor":      "Profit Factor",
    "expectancy_currency": "Expectancy (currency)",
    "expectancy_r":       "Expectancy (R)",
    "max_drawdown":       "Max Drawdown",
    "trade_count":        "Trade Count",
}


def _fmt_stat(key: str, value: object) -> str:
    """Format a stat value for display in the stats table."""
    if key == "win_rate":
        return f"{float(value) * 100:.1f}%"
    if key == "trade_count":
        return str(int(value))
    if key == "profit_factor":
        v = float(value)
        return "∞" if v == float("inf") else f"{v:.2f}"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


class SummaryWindow(QtWidgets.QDialog):
    """Three-tab session summary dialog.

    Parameters
    ----------
    session_id:
        UUID of the session to display.
    trade_store:
        Open :class:`~backtester.store.db.TradeStore` for the session.
    parent:
        Optional parent widget.
    """

    def __init__(self, session_id: str, trade_store, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._session_id  = session_id
        self._trade_store = trade_store

        self.setWindowTitle(f"Session Summary — {session_id[:8]}")
        self.resize(900, 600)
        self.setStyleSheet(f"background: {_BG}; color: {_FG};")

        trades = trade_store.get_trades_for_session(session_id)

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid #2a2e39; background: {_BG}; }}"
            f"QTabBar::tab {{ background: {_HDR_BG}; color: {_FG}; "
            f"  padding: 6px 18px; border: 1px solid #2a2e39; }}"
            f"QTabBar::tab:selected {{ background: #1e3a5f; }}"
        )
        tabs.addTab(self._build_equity_tab(trades), "Equity Curve")
        tabs.addTab(self._build_stats_tab(trades),  "Stats")
        tabs.addTab(self._build_log_tab(trades),    "Trade Log")

        btn_export = QtWidgets.QPushButton("Export…")
        btn_export.setStyleSheet(
            f"QPushButton {{ background: #252932; color: {_FG}; border: 1px solid #363a45;"
            f"  padding: 5px 16px; border-radius: 3px; }}"
            f"QPushButton:hover {{ background: #363a45; }}"
        )
        btn_export.clicked.connect(self._export)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setStyleSheet(btn_export.styleSheet())
        btn_close.clicked.connect(self.accept)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)
        vbox.addWidget(tabs)
        vbox.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_equity_tab(self, trades: list) -> QtWidgets.QWidget:
        """PyQtGraph line chart of cumulative P&L."""
        plot = pg.PlotWidget(background=_BG)
        plot.setMenuEnabled(False)
        plot.setLabel("bottom", "Trade #")
        plot.setLabel("left",   "Cumulative P&L")
        plot.getAxis("bottom").setPen(pg.mkPen(_DIM))
        plot.getAxis("left").setPen(pg.mkPen(_DIM))

        curve = compute_equity_curve(trades)
        if len(curve) > 0:
            final  = float(curve.iloc[-1])
            colour = _GREEN if final >= 0 else _RED
            x = list(curve.index)
            y = list(curve.values)
            plot.plot(x, y, pen=pg.mkPen(colour, width=2))

        # Zero line
        zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(_DIM, style=QtCore.Qt.PenStyle.DashLine))
        plot.addItem(zero)

        w = QtWidgets.QWidget()
        w.setStyleSheet(f"background: {_BG};")
        box = QtWidgets.QVBoxLayout(w)
        box.setContentsMargins(4, 4, 4, 4)
        box.addWidget(plot)
        return w

    def _build_stats_tab(self, trades: list) -> QtWidgets.QWidget:
        """Two-column QTableWidget: Metric | Value."""
        stats = compute_summary_stats(trades)

        tbl = QtWidgets.QTableWidget(len(_DISPLAY_NAMES), 2)
        tbl.setStyleSheet(_TABLE_CSS)
        tbl.setHorizontalHeaderLabels(["Metric", "Value"])
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setAlternatingRowColors(True)

        for row, (key, label) in enumerate(_DISPLAY_NAMES.items()):
            tbl.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            tbl.setItem(row, 1, QtWidgets.QTableWidgetItem(_fmt_stat(key, stats.get(key, 0))))

        tbl.resizeColumnsToContents()

        w = QtWidgets.QWidget()
        w.setStyleSheet(f"background: {_BG};")
        box = QtWidgets.QVBoxLayout(w)
        box.setContentsMargins(4, 4, 4, 4)
        box.addWidget(tbl)
        return w

    def _build_log_tab(self, trades: list) -> QtWidgets.QWidget:
        """Sortable 9-column trade log table."""
        cols = ["#", "Entry DT", "Exit DT", "Side", "Entry", "Exit", "P&L", "R", "Reason"]
        tbl  = QtWidgets.QTableWidget(len(trades), len(cols))
        tbl.setStyleSheet(_TABLE_CSS)
        tbl.setHorizontalHeaderLabels(cols)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        # Sorting must be disabled during population — enabling it before setItem()
        # causes Qt to re-sort after each call, scrambling row positions.
        tbl.setSortingEnabled(False)
        tbl.setAlternatingRowColors(True)

        for row, t in enumerate(trades):
            r_str = f"{t.pnl_r:.2f}" if t.pnl_r is not None else "—"
            row_data = [
                str(row + 1),
                str(t.entry_datetime),
                str(t.exit_datetime),
                t.side,
                f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}",
                f"{t.pnl_currency:.2f}",
                r_str,
                t.exit_reason or "—",
            ]
            for col, val in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(val)
                # Numeric columns: right-align
                if col in (0, 4, 5, 6, 7):
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                tbl.setItem(row, col, item)

            # Row color: green tint for winners, red tint for losers
            if t.pnl_currency > 0:
                row_color = QtGui.QColor(52, 211, 153, 38)    # #34d399 ~15% alpha
            elif t.pnl_currency < 0:
                row_color = QtGui.QColor(248, 113, 113, 38)   # #f87171 ~15% alpha
            else:
                row_color = None
            if row_color is not None:
                for col in range(len(cols)):
                    item = tbl.item(row, col)
                    if item is not None:
                        item.setBackground(row_color)

        tbl.resizeColumnsToContents()
        # Re-enable sorting after all data is in place
        tbl.setSortingEnabled(True)

        w = QtWidgets.QWidget()
        w.setStyleSheet(f"background: {_BG};")
        box = QtWidgets.QVBoxLayout(w)
        box.setContentsMargins(4, 4, 4, 4)
        box.addWidget(tbl)
        return w

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self) -> None:
        dir_ = QtWidgets.QFileDialog.getExistingDirectory(self, "Export to…")
        if not dir_:
            return
        stem = self._session_id[:8]
        self._trade_store.export_csv(
            self._session_id, f"{dir_}/{stem}_trades.csv"
        )
        self._trade_store.export_equity_curve(
            self._session_id, f"{dir_}/{stem}_equity.csv"
        )
        QtWidgets.QMessageBox.information(
            self, "Exported", f"Files written to {dir_}"
        )
