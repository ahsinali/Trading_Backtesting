"""OrderPanel — order entry + live trade log for the current session."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

from backtester.engine.cursor import BarCursor
from backtester.sim.orders import Order, snap_to_tick

# TradeStore imported lazily to avoid circular-import risk at module level
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backtester.store.db import TradeStore

# ── Style constants ───────────────────────────────────────────────────────────
_BG      = "#1e222d"
_FG      = "#c8cdd4"
_DIM     = "#555d6b"
_GREEN   = "#26a69a"
_RED     = "#ef5350"
_HOVER   = "#2a2e39"
_HDR_BG  = "#131722"

_DIM_CSS = f"color: {_DIM}; font-size: 11px; background: transparent;"

_COMBO_CSS = f"""
    QComboBox {{
        color: {_FG};
        background: #252932;
        border: 1px solid {_DIM};
        border-radius: 3px;
        padding: 2px 6px;
        font-size: 12px;
    }}
    QComboBox:disabled {{
        color: {_DIM};
        background: #1a1e27;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 16px;
    }}
    QComboBox QAbstractItemView {{
        background: #252932;
        color: {_FG};
        selection-background-color: #2a2e39;
        border: 1px solid {_DIM};
    }}
"""

_SPIN_CSS = f"""
    QDoubleSpinBox {{
        color: {_FG};
        background: #252932;
        border: 1px solid {_DIM};
        border-radius: 3px;
        padding: 2px 4px;
        font-size: 12px;
    }}
"""

_TABLE_CSS = f"""
    QTableWidget {{
        background: {_HDR_BG};
        color: {_FG};
        border: none;
        font-size: 11px;
        gridline-color: #252932;
    }}
    QTableWidget::item {{
        padding: 2px 4px;
    }}
    QTableWidget::item:selected {{
        background: #2a2e39;
        color: {_FG};
    }}
    QHeaderView::section {{
        background: #252932;
        color: {_DIM};
        font-size: 10px;
        padding: 3px 4px;
        border: none;
        border-right: 1px solid #131722;
        border-bottom: 1px solid #131722;
    }}
    QScrollBar:vertical {{
        background: {_HDR_BG};
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background: #363a45;
        border-radius: 4px;
    }}
"""

_LONG_BTN = """
    QPushButton {
        background: #26a69a;
        color: #ffffff;
        font-weight: bold;
        font-size: 9px;
        border: none;
        border-radius: 0px;
        min-height: 36px;
    }
    QPushButton:hover   { background: #00796b; }
    QPushButton:pressed { background: #004d40; }
    QPushButton:disabled {
        background: #26a69a;
        color: #ffffff;
        opacity: 0.4;
    }
"""

_SHORT_BTN = """
    QPushButton {
        background: #ef5350;
        color: #ffffff;
        font-weight: bold;
        font-size: 9px;
        border: none;
        border-radius: 0px;
        min-height: 36px;
    }
    QPushButton:hover   { background: #c62828; }
    QPushButton:pressed { background: #b71c1c; }
    QPushButton:disabled {
        background: #ef5350;
        color: #ffffff;
        opacity: 0.4;
    }
"""

_FLATTEN_BTN_ENABLED = """
    QPushButton {
        background: #334155;
        color: #94a3b8;
        font-weight: bold;
        font-size: 11px;
        border: none;
        border-radius: 0px;
        min-height: 36px;
    }
    QPushButton:hover   { background: #475569; color: #e2e8f0; }
    QPushButton:pressed { background: #1e293b; }
"""

_FLATTEN_BTN_DISABLED = """
    QPushButton {
        background: #1e2530;
        color: #3d4a5c;
        font-weight: bold;
        font-size: 11px;
        border: none;
        border-radius: 0px;
        min-height: 36px;
    }
"""

# Column definitions: (header, min-width)
_COLS = [
    ("#",       28),
    ("Side",    42),
    ("Entry",   62),
    ("Qty",     44),
    ("Stop",    62),
    ("Target",  62),
    ("Exit",    62),
    ("P&L",     62),
    ("R",       42),
]


def _dim_label(text: str, parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text, parent)
    lbl.setStyleSheet(_DIM_CSS)
    return lbl


class OrderPanel(QtWidgets.QWidget):
    """Combined order-entry bar and live trade-log panel.

    Signals
    -------
    order_placed(Order)
        Emitted when the user clicks Long or Short.
    policy_changed(str)
        Emitted when the ambiguity policy combo changes (lowercase name).
    flatten_requested()
        Emitted when the user clicks the Flatten button.
    """

    order_placed:      Signal = Signal(object)
    policy_changed:    Signal = Signal(str)   # emits lowercase policy name
    flatten_requested: Signal = Signal()

    def __init__(
        self,
        cursor:      BarCursor,
        session_id:  str,
        tick_size:   float = 0.01,
        trade_store: "TradeStore | None" = None,
        parent:      QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cursor      = cursor
        self._session_id  = session_id
        self._tick_size   = tick_size
        self._trade_store = trade_store

        # Price defaults updated by MainWindow on each bar advance
        self._last_close: float = 0.0
        self._last_atr:   float = 0.0

        self.setStyleSheet(f"background: {_BG};")

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Input row ─────────────────────────────────────────────────
        vbox.addWidget(self._build_input_row())

        # ── Separator ─────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #252932;")
        vbox.addWidget(sep)

        # ── Action bar (Long / Short / Flatten) ───────────────────────
        vbox.addWidget(self._build_action_bar())

        # ── Separator ─────────────────────────────────────────────────
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #252932;")
        vbox.addWidget(sep2)

        # ── Trade log table ───────────────────────────────────────────
        self._table = self._build_table()
        vbox.addWidget(self._table)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def policy(self) -> str:
        """Current ambiguity policy as a lowercase string."""
        return self._combo_policy.currentText().lower()

    def lock_policy(self) -> None:
        """Prevent policy changes after the first order is placed."""
        self._combo_policy.setEnabled(False)

    def set_trade_store(self, store: "TradeStore") -> None:
        """Attach (or replace) the :class:`~backtester.store.db.TradeStore`."""
        self._trade_store = store

    def update_price_defaults(self, close: float, atr: float) -> None:
        """Called by MainWindow after each bar advance to keep price inputs current."""
        self._last_close = close
        self._last_atr   = atr

    def update_button_states(self, has_pending: bool, has_open_trade: bool) -> None:
        """Reflect the current position state in the action button bar.

        Parameters
        ----------
        has_pending:
            ``True`` when there is at least one unfilled entry order.
        has_open_trade:
            ``True`` when FillEngine has at least one open position.
        """
        can_enter = not has_pending and not has_open_trade
        self._btn_long.setEnabled(can_enter)
        self._btn_short.setEnabled(can_enter)

        flatten_on = has_open_trade and not has_pending
        self._btn_flatten.setEnabled(flatten_on)
        self._btn_flatten.setStyleSheet(
            _FLATTEN_BTN_ENABLED if flatten_on else _FLATTEN_BTN_DISABLED
        )

        self.pending_label.setVisible(has_pending)
        if not has_pending:
            self.pending_label.setText("")

    def refresh_trade_log(self) -> None:
        """Query the store for the current session and repopulate the table."""
        self._table.setRowCount(0)
        if self._trade_store is None:
            return

        trades = self._trade_store.get_trades_for_session(self._session_id)
        self._table.setRowCount(len(trades))

        for row_idx, t in enumerate(trades):
            pnl    = t.pnl_currency
            colour = QtGui.QColor(_GREEN if pnl >= 0 else _RED)

            def _cell(text: str, align=QtCore.Qt.AlignmentFlag.AlignRight) -> QtWidgets.QTableWidgetItem:
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(align | QtCore.Qt.AlignmentFlag.AlignVCenter)
                item.setForeground(colour)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                return item

            num_item = QtWidgets.QTableWidgetItem(str(row_idx + 1))
            num_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            num_item.setForeground(QtGui.QColor(_DIM))
            num_item.setFlags(num_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 0, num_item)

            side_colour = QtGui.QColor(_GREEN if t.side == "long" else _RED)
            side_item = QtWidgets.QTableWidgetItem(t.side.capitalize())
            side_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            side_item.setForeground(side_colour)
            side_item.setFlags(side_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, 1, side_item)

            self._table.setItem(row_idx, 2, _cell(f"{t.entry_price:.2f}"))
            self._table.setItem(row_idx, 3, _cell(f"{t.quantity:.2f}"))
            self._table.setItem(row_idx, 4, _cell(
                f"{t.stop_price:.2f}"   if t.stop_price   is not None else "—"
            ))
            self._table.setItem(row_idx, 5, _cell(
                f"{t.target_price:.2f}" if t.target_price is not None else "—"
            ))
            self._table.setItem(row_idx, 6, _cell(f"{t.exit_price:.2f}"))

            sign = "+" if pnl >= 0 else ""
            self._table.setItem(row_idx, 7, _cell(f"{sign}{pnl:.2f}"))

            r_text = f"{t.pnl_r:+.2f}R" if t.pnl_r is not None else "—"
            self._table.setItem(row_idx, 8, _cell(r_text))

        if trades:
            self._table.scrollToBottom()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_input_row(self) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget()
        row_widget.setFixedHeight(44)
        row_widget.setStyleSheet(f"background: {_BG};")

        row = QtWidgets.QHBoxLayout(row_widget)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(8)

        # Order type
        row.addWidget(_dim_label("Type:", row_widget))
        self._combo_type = QtWidgets.QComboBox()
        self._combo_type.addItems(["Market", "Limit", "Stop"])
        self._combo_type.setFixedWidth(76)
        self._combo_type.setStyleSheet(_COMBO_CSS)
        self._combo_type.setToolTip(
            "Market: fill at next bar open\n"
            "Limit:  fill when price touches the limit level\n"
            "Stop:   fill when price breaks through the stop trigger"
        )
        self._combo_type.currentTextChanged.connect(self._on_type_changed)
        row.addWidget(self._combo_type)

        # Price input — hidden until Limit or Stop is selected
        self._sep_price  = _dim_label("│", row_widget)
        self._lbl_price  = _dim_label("Price:", row_widget)
        self._spin_price = QtWidgets.QDoubleSpinBox()
        self._spin_price.setRange(0.0, 1_000_000)
        self._spin_price.setValue(0.0)
        self._spin_price.setDecimals(2)
        self._spin_price.setFixedWidth(90)
        self._spin_price.setStyleSheet(_SPIN_CSS)
        row.addWidget(self._sep_price)
        row.addWidget(self._lbl_price)
        row.addWidget(self._spin_price)
        self._sep_price.setVisible(False)
        self._lbl_price.setVisible(False)
        self._spin_price.setVisible(False)

        row.addWidget(_dim_label("│", row_widget))

        # Qty
        row.addWidget(_dim_label("Qty:", row_widget))
        self._spin_qty = QtWidgets.QDoubleSpinBox()
        self._spin_qty.setRange(0.01, 100_000)
        self._spin_qty.setValue(1.0)
        self._spin_qty.setDecimals(2)
        self._spin_qty.setFixedWidth(72)
        self._spin_qty.setStyleSheet(_SPIN_CSS)
        row.addWidget(self._spin_qty)

        row.addWidget(_dim_label("│", row_widget))

        # Stop Δ
        row.addWidget(_dim_label("Stop Δ:", row_widget))
        self._spin_stop = QtWidgets.QDoubleSpinBox()
        self._spin_stop.setRange(0.0, 1_000_000)
        self._spin_stop.setValue(0.0)
        self._spin_stop.setDecimals(2)
        self._spin_stop.setFixedWidth(72)
        self._spin_stop.setStyleSheet(_SPIN_CSS)
        self._spin_stop.setToolTip("Price offset from entry for stop-loss (0 = no stop)")
        row.addWidget(self._spin_stop)

        # Target Δ
        row.addWidget(_dim_label("Target Δ:", row_widget))
        self._spin_target = QtWidgets.QDoubleSpinBox()
        self._spin_target.setRange(0.0, 1_000_000)
        self._spin_target.setValue(0.0)
        self._spin_target.setDecimals(2)
        self._spin_target.setFixedWidth(72)
        self._spin_target.setStyleSheet(_SPIN_CSS)
        self._spin_target.setToolTip("Price offset from entry for take-profit (0 = no target)")
        row.addWidget(self._spin_target)

        row.addWidget(_dim_label("│", row_widget))

        # Ambiguity policy
        row.addWidget(_dim_label("Policy:", row_widget))
        self._combo_policy = QtWidgets.QComboBox()
        self._combo_policy.addItems(["Default", "Conservative", "Optimistic", "Unknown"])
        self._combo_policy.setFixedWidth(110)
        self._combo_policy.setStyleSheet(_COMBO_CSS)
        self._combo_policy.setToolTip(
            "Ambiguity policy — locked after the first order is placed.\n"
            "Default: time-priority (OHLC order)\n"
            "Conservative: stop wins (worst case)\n"
            "Optimistic: target wins (best case)\n"
            "Unknown: defer until unambiguous bar"
        )
        self._combo_policy.currentTextChanged.connect(self._on_policy_changed)
        row.addWidget(self._combo_policy)

        row.addStretch()
        return row_widget

    def _build_action_bar(self) -> QtWidgets.QWidget:
        """Build the pending indicator label + Short / Qty / Long / Flatten bar."""
        container = QtWidgets.QWidget()
        container.setStyleSheet(f"background: {_BG};")

        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # ── Pending indicator label ────────────────────────────────────
        self.pending_label = QtWidgets.QLabel("")
        self.pending_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pending_label.setFixedHeight(20)
        self.pending_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        self.pending_label.setVisible(False)
        vbox.addWidget(self.pending_label)

        # ── Button bar ────────────────────────────────────────────────
        btn_row = QtWidgets.QWidget()
        btn_row.setStyleSheet(f"background: {_BG};")
        hbox = QtWidgets.QHBoxLayout(btn_row)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)

        # Short button
        self._btn_short = QtWidgets.QPushButton("Short")
        self._btn_short.setStyleSheet(_SHORT_BTN)
        self._btn_short.setFixedWidth(60)
        self._btn_short.setFixedHeight(36)
        self._btn_short.clicked.connect(self._on_short_clicked)
        hbox.addWidget(self._btn_short)

        # Qty display
        self._lbl_qty_display = QtWidgets.QLabel("1.00")
        self._lbl_qty_display.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._lbl_qty_display.setFixedWidth(48)
        self._lbl_qty_display.setFixedHeight(36)
        self._lbl_qty_display.setStyleSheet(
            "background: #1e2530;"
            "color: #94a3b8;"
            "border-left: 1px solid rgba(255,255,255,0.08);"
            "border-right: 1px solid rgba(255,255,255,0.08);"
            "font-size: 11px;"
        )
        hbox.addWidget(self._lbl_qty_display)

        # Long button
        self._btn_long = QtWidgets.QPushButton("Long")
        self._btn_long.setStyleSheet(_LONG_BTN)
        self._btn_long.setFixedWidth(60)
        self._btn_long.setFixedHeight(36)
        self._btn_long.clicked.connect(self._on_long_clicked)
        hbox.addWidget(self._btn_long)

        # Small gap before Flatten
        hbox.addSpacing(4)

        # Flatten button
        self._btn_flatten = QtWidgets.QPushButton("Flatten")
        self._btn_flatten.setStyleSheet(_FLATTEN_BTN_DISABLED)
        self._btn_flatten.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._btn_flatten.setFixedHeight(36)
        self._btn_flatten.setEnabled(False)
        self._btn_flatten.clicked.connect(self._on_flatten_clicked)
        hbox.addWidget(self._btn_flatten)

        vbox.addWidget(btn_row)

        # Keep qty display in sync with spinbox
        self._spin_qty.valueChanged.connect(
            lambda v: self._lbl_qty_display.setText(f"{v:.2f}")
        )

        return container

    def _build_table(self) -> QtWidgets.QTableWidget:
        headers = [c[0] for c in _COLS]
        table   = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setStyleSheet(_TABLE_CSS)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        table.verticalHeader().setVisible(False)
        table.setShowGrid(True)
        table.setAlternatingRowColors(False)
        table.setMinimumHeight(120)

        hdr = table.horizontalHeader()
        for col, (_, min_w) in enumerate(_COLS):
            table.setColumnWidth(col, min_w)
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(8, QtWidgets.QHeaderView.ResizeMode.Stretch)

        return table

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_type_changed(self, text: str) -> None:
        is_price = text in ("Limit", "Stop")
        self._sep_price.setVisible(is_price)
        self._lbl_price.setVisible(is_price)
        self._spin_price.setVisible(is_price)
        if is_price:
            self._lbl_price.setText("Limit Price:" if text == "Limit" else "Stop Price:")
            self._repopulate_price_spin()

    def _on_policy_changed(self, text: str) -> None:
        self.policy_changed.emit(text.lower())

    def _on_long_clicked(self) -> None:
        if not self._btn_long.isEnabled():
            QtWidgets.QMessageBox.warning(
                self, "Order Pending",
                "An order is already pending. Cancel it before placing a new one."
            )
            return
        order = self._build_order("long")
        self._show_pending_indicator("long")
        self.lock_policy()
        self.order_placed.emit(order)

    def _on_short_clicked(self) -> None:
        if not self._btn_short.isEnabled():
            QtWidgets.QMessageBox.warning(
                self, "Order Pending",
                "An order is already pending. Cancel it before placing a new one."
            )
            return
        order = self._build_order("short")
        self._show_pending_indicator("short")
        self.lock_policy()
        self.order_placed.emit(order)

    def _on_flatten_clicked(self) -> None:
        self.flatten_requested.emit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _show_pending_indicator(self, side: str) -> None:
        if side == "long":
            text   = "⏳ LONG PENDING — waiting for fill"
            colour = "#26a69a"
        else:
            text   = "⏳ SHORT PENDING — waiting for fill"
            colour = "#ef5350"
        self.pending_label.setText(text)
        self.pending_label.setStyleSheet(
            f"font-size: 10px; font-weight: bold; color: {colour};"
        )
        self.pending_label.setVisible(True)

    def _repopulate_price_spin(self) -> None:
        """Set the price spin to a sensible default for the current type."""
        self._spin_price.setValue(self._last_close)

    def _build_order(self, side: str) -> Order:
        """Package current UI values into an :class:`~backtester.sim.orders.Order`."""
        bar        = self._cursor.visible_bars.iloc[-1]
        qty        = self._spin_qty.value()
        raw_stop   = self._spin_stop.value()
        raw_target = self._spin_target.value()
        order_type = self._combo_type.currentText().lower()

        if order_type == "market":
            ref_price = snap_to_tick(float(bar["close"]), self._tick_size)
            stop, target = self._compute_bracket(ref_price, side, raw_stop, raw_target)
            return Order.market(
                session_id       = self._session_id,
                side             = side,
                quantity         = qty,
                tick_size        = self._tick_size,
                created_at_index = self._cursor.current_index,
                stop_price       = stop,
                target_price     = target,
            )

        elif order_type == "limit":
            ref_price = snap_to_tick(self._spin_price.value(), self._tick_size)
            stop, target = self._compute_bracket(ref_price, side, raw_stop, raw_target)
            return Order.limit(
                session_id       = self._session_id,
                side             = side,
                quantity         = qty,
                limit_price      = ref_price,
                tick_size        = self._tick_size,
                created_at_index = self._cursor.current_index,
                stop_price       = stop,
                target_price     = target,
            )

        else:  # stop
            ref_price = snap_to_tick(self._spin_price.value(), self._tick_size)
            bracket_stop, bracket_target = self._compute_bracket(
                ref_price, side, raw_stop, raw_target
            )
            return Order.stop_entry(
                session_id       = self._session_id,
                side             = side,
                quantity         = qty,
                stop_price       = ref_price,
                tick_size        = self._tick_size,
                created_at_index = self._cursor.current_index,
                bracket_stop     = bracket_stop,
                bracket_target   = bracket_target,
            )

    def _compute_bracket(
        self,
        ref_price:  float,
        side:       str,
        raw_stop:   float,
        raw_target: float,
    ) -> tuple[float | None, float | None]:
        """Compute absolute bracket stop/target from offsets relative to *ref_price*."""
        if side == "long":
            stop   = snap_to_tick(ref_price - raw_stop,   self._tick_size) if raw_stop   > 0 else None
            target = snap_to_tick(ref_price + raw_target, self._tick_size) if raw_target > 0 else None
        else:
            stop   = snap_to_tick(ref_price + raw_stop,   self._tick_size) if raw_stop   > 0 else None
            target = snap_to_tick(ref_price - raw_target, self._tick_size) if raw_target > 0 else None
        return stop, target
