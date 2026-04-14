"""MainWindow — top-level application window with TradingView-style info bar."""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from backtester.engine.cursor import BarCursor
from backtester.io.loader import load_csv
from backtester.sim.fills import FillEngine
from backtester.sim.orders import Order, Trade
from backtester.store.db import TradeStore
from backtester.store.manifest import SessionManifest
from backtester.ui.chart_widget import ChartWidget
from backtester.ui.hotkeys import install_hotkeys
from backtester.ui.order_panel import OrderPanel

# Project root: backtester/ui/main_window.py → ../../.. = TrdBcktest/
_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# ── style constants ───────────────────────────────────────────────────────────
_BAR_BG      = "#1e222d"   # info-bar background
_BAR_FG      = "#c8cdd4"   # default label foreground
_BAR_DIM     = "#555d6b"   # dimmed / separator colour
_BTN_HOVER   = "#2a2e39"   # button hover background
_GREEN       = "#26a69a"
_RED         = "#ef5350"

_LABEL_CSS = f"""
    QLabel {{
        color: {_BAR_FG};
        font-family: 'Segoe UI', 'SF Pro Text', sans-serif;
        font-size: 12px;
        background: transparent;
    }}
"""

_BTN_CSS = f"""
    QPushButton {{
        color: {_BAR_FG};
        background: transparent;
        border: none;
        font-family: 'Segoe UI', 'SF Pro Text', sans-serif;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background: {_BTN_HOVER};
    }}
    QPushButton:pressed {{
        background: #363a45;
    }}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_timeframe(bars: pd.DataFrame) -> str:
    """Best-effort timeframe detection from bar spacing."""
    if len(bars) < 2:
        return "—"
    delta = int((bars.index[1] - bars.index[0]).total_seconds())
    mapping = {
        60:    "1m",
        300:   "5m",
        900:   "15m",
        1800:  "30m",
        3600:  "1h",
        14400: "4h",
        86400: "1D",
        604800:"1W",
    }
    return mapping.get(delta, f"{delta // 60}m" if delta < 86400 else f"{delta // 86400}D")


def _sep(parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
    """Thin vertical separator label."""
    lbl = QtWidgets.QLabel("│", parent)
    lbl.setStyleSheet(f"color: {_BAR_DIM}; font-size: 11px; background: transparent;")
    return lbl


# ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    """Application shell with TradingView-style top info bar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Backtester")
        self.resize(1400, 800)
        self._set_dark_theme()

        self._cursor: BarCursor | None = None
        self._chart:  ChartWidget | None = None

        # Sim layer (created fresh for each CSV)
        self._session_id:    str                 = ""
        self._fill_engine:   FillEngine | None   = None
        self._trade_store:   TradeStore | None   = None
        self._pending_orders: list[Order]        = []
        self._order_panel:   OrderPanel | None   = None
        self._manifest:      SessionManifest | None = None

        # True while in review mode — prevents Space from advancing bars
        self._advance_paused: bool = False

        # Placeholders filled by _open_csv
        self._lbl_symbol:    QtWidgets.QLabel | None = None
        self._lbl_tf:        QtWidgets.QLabel | None = None
        self._lbl_o:         QtWidgets.QLabel | None = None
        self._lbl_h:         QtWidgets.QLabel | None = None
        self._lbl_l:         QtWidgets.QLabel | None = None
        self._lbl_c:         QtWidgets.QLabel | None = None
        self._lbl_delta:     QtWidgets.QLabel | None = None
        self._btn_ind:       QtWidgets.QPushButton | None = None
        self._btn_fit:       QtWidgets.QPushButton | None = None
        self._btn_prev:      QtWidgets.QPushButton | None = None
        self._review_banner: QtWidgets.QLabel | None = None
        self._review_act:    QtWidgets.QAction | None = None
        self._summary_act:   QtWidgets.QAction | None = None

        self._build_menu()

        # Placeholder central widget shown before a file is loaded
        splash = QtWidgets.QLabel(
            "Open a CSV file to begin  (File > Open CSV  or  Ctrl+O)",
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        splash.setStyleSheet(f"color: {_BAR_DIM}; font-size: 14px; background: #131722;")
        self.setCentralWidget(splash)

        self.statusBar().setStyleSheet(
            f"color: {_BAR_DIM}; background: {_BAR_BG}; font-size: 11px;"
        )
        self.statusBar().showMessage("No file loaded — File > Open CSV")

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        menubar.setStyleSheet(
            f"QMenuBar {{ background: {_BAR_BG}; color: {_BAR_FG}; }}"
            f"QMenuBar::item:selected {{ background: {_BTN_HOVER}; }}"
            f"QMenu {{ background: {_BAR_BG}; color: {_BAR_FG}; border: 1px solid #2a2e39; }}"
            f"QMenu::item:selected {{ background: {_BTN_HOVER}; }}"
        )
        file_menu = menubar.addMenu("&File")

        open_act = file_menu.addAction("&Open CSV…")
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_csv)

        file_menu.addSeparator()

        quit_act = file_menu.addAction("&Quit")
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)

        view_menu = menubar.addMenu("&View")
        self._review_act = view_menu.addAction("&Review Mode")
        self._review_act.setCheckable(True)
        self._review_act.setEnabled(False)   # enabled after first bar advance
        self._review_act.triggered.connect(self._toggle_review_mode)

        session_menu = menubar.addMenu("&Session")
        self._summary_act = session_menu.addAction("View &Summary…")
        self._summary_act.setEnabled(False)   # enabled once there are trades
        self._summary_act.triggered.connect(self._show_summary)

    # ------------------------------------------------------------------
    # Info bar
    # ------------------------------------------------------------------

    def _build_info_bar(self, symbol: str, timeframe: str) -> QtWidgets.QWidget:
        """Construct the TradingView-style top info bar."""
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet(f"background: {_BAR_BG};")

        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(6)

        # Symbol
        self._lbl_symbol = QtWidgets.QLabel(symbol)
        self._lbl_symbol.setStyleSheet(
            f"color: {_BAR_FG}; font-size: 13px; font-weight: 600; background: transparent;"
        )
        row.addWidget(self._lbl_symbol)

        # Timeframe
        self._lbl_tf = QtWidgets.QLabel(timeframe)
        self._lbl_tf.setStyleSheet(
            f"color: {_BAR_DIM}; font-size: 12px; background: transparent;"
        )
        row.addWidget(self._lbl_tf)

        row.addWidget(_sep(bar))

        # O H L C labels
        for attr, tag in [
            ("_lbl_o", "O"),
            ("_lbl_h", "H"),
            ("_lbl_l", "L"),
            ("_lbl_c", "C"),
        ]:
            tag_lbl = QtWidgets.QLabel(f"{tag}:")
            tag_lbl.setStyleSheet(f"color: {_BAR_DIM}; font-size: 11px; background: transparent;")
            row.addWidget(tag_lbl)

            val_lbl = QtWidgets.QLabel("—")
            val_lbl.setStyleSheet(_LABEL_CSS)
            setattr(self, attr, val_lbl)
            row.addWidget(val_lbl)

        row.addWidget(_sep(bar))

        # Δ change
        self._lbl_delta = QtWidgets.QLabel("—")
        self._lbl_delta.setStyleSheet(_LABEL_CSS)
        row.addWidget(self._lbl_delta)

        row.addStretch()

        # Indicator toggle button
        self._btn_ind = QtWidgets.QPushButton("Indicators  (I)")
        self._btn_ind.setStyleSheet(_BTN_CSS)
        self._btn_ind.clicked.connect(self._toggle_indicators)
        row.addWidget(self._btn_ind)

        # Fit button
        self._btn_fit = QtWidgets.QPushButton("Fit  (F)")
        self._btn_fit.setStyleSheet(_BTN_CSS)
        self._btn_fit.clicked.connect(self._fit_chart)
        row.addWidget(self._btn_fit)

        # Prev button (step back in review mode)
        self._btn_prev = QtWidgets.QPushButton("◀ Prev")
        self._btn_prev.setStyleSheet(_BTN_CSS)
        self._btn_prev.clicked.connect(self._step_back)
        row.addWidget(self._btn_prev)

        return bar

    # ------------------------------------------------------------------
    # File open
    # ------------------------------------------------------------------

    def _open_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open OHLCV CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            bars = load_csv(path)
        except (FileNotFoundError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to load CSV", str(exc))
            return

        self._cursor = BarCursor(bars)

        # Pre-load the first 20 bars so the chart opens at a readable zoom level.
        # Advance silently; the ChartWidget constructor will render them all.
        n_preload = min(19, len(bars) - 1)
        for _ in range(n_preload):
            self._cursor.advance()

        symbol    = Path(path).stem
        timeframe = _detect_timeframe(bars)

        # ── Sim layer ──────────────────────────────────────────────────
        self._session_id     = str(uuid.uuid4())
        self._pending_orders = []
        self._fill_engine    = FillEngine(
            tick_size   = 0.01,
            symbol      = symbol,
            timeframe   = timeframe,
            config_hash = self._session_id,
        )
        # One DB file per session, stored in the project root so that
        # `find ~/claudeCode/TrdBcktest -name "*.db"` can locate it.
        db_path = str(_PROJECT_DIR / f"{self._session_id}.db")
        if self._trade_store is not None:
            self._trade_store.close()
        self._trade_store = TradeStore(db_path)   # create_tables() called inside

        # ── UI ─────────────────────────────────────────────────────────
        self._chart = ChartWidget(self._cursor, parent=self)

        info_bar = self._build_info_bar(symbol, timeframe)

        # Review Mode banner (hidden until activated)
        self._review_banner = QtWidgets.QLabel(
            "  ◀  REVIEW MODE — View > Review Mode to resume session  "
        )
        self._review_banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._review_banner.setStyleSheet(
            "background: #f9a825; color: #1a1a1a; font-weight: 600;"
            " font-size: 12px; padding: 4px;"
        )
        self._review_banner.setVisible(False)

        # Order panel — receives trade_store so refresh_trade_log() can query it
        self._order_panel = OrderPanel(
            cursor      = self._cursor,
            session_id  = self._session_id,
            tick_size   = 0.01,
            trade_store = self._trade_store,
            parent      = self,
        )
        self._order_panel.order_placed.connect(self._on_order_placed)
        self._order_panel.policy_changed.connect(self._on_policy_changed)

        # Chart + order panel share vertical space via a splitter so the
        # trade log is always visible and resizable.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setStyleSheet("QSplitter::handle { background: #252932; height: 3px; }")
        splitter.addWidget(self._chart)
        splitter.addWidget(self._order_panel)
        splitter.setSizes([580, 180])          # initial split: chart gets most space
        splitter.setChildrenCollapsible(False)

        # Container: info bar → review banner → splitter
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(info_bar)
        vbox.addWidget(self._review_banner)
        vbox.addWidget(splitter, 1)            # splitter stretches
        self.setCentralWidget(container)

        self.setWindowTitle(f"Backtester — {symbol}")
        if self._review_act is not None:
            self._review_act.setEnabled(False)
            self._review_act.setChecked(False)
        if self._summary_act is not None:
            self._summary_act.setEnabled(False)

        # Save session manifest
        self._manifest = SessionManifest(
            session_id           = self._session_id,
            symbol               = symbol,
            timeframe            = timeframe,
            bar_range            = (str(bars.index[0]), str(bars.index[-1])),
            data_checksum        = bars.attrs.get("checksum", ""),
            indicator_config     = {
                "bb_period": 20, "sma_period": 20,
                "rsi_period": 14, "atr_period": 14,
            },
            anonymization_config = None,
            created_at           = datetime.datetime.utcnow().isoformat(),
        )
        self._manifest.config_hash = self._manifest.compute_config_hash()
        manifest_path = str(_PROJECT_DIR / f"{self._session_id}.manifest.json")
        self._manifest.save(manifest_path)

        install_hotkeys(self, self._cursor, self._chart)
        self.after_bar_advance()

    # ------------------------------------------------------------------
    # Button callbacks (so buttons work in addition to hotkeys)
    # ------------------------------------------------------------------

    def _toggle_indicators(self) -> None:
        if self._chart:
            self._chart.toggle_indicators()

    def _fit_chart(self) -> None:
        if self._chart:
            self._chart.fit_to_visible()

    def _set_advance_enabled(self, enabled: bool) -> None:
        """Enable or disable forward bar-advance controls.

        Called when entering/exiting review mode so that Space and the
        hotkey handler cannot advance past the current bar while reviewing.
        """
        self._advance_paused = not enabled

    def _toggle_review_mode(self) -> None:
        if self._cursor is None:
            return
        try:
            if self._cursor.in_review_mode:
                self._cursor.exit_review_mode()
                if self._review_banner is not None:
                    self._review_banner.setVisible(False)
                if self._review_act is not None:
                    self._review_act.setChecked(False)
                if self._btn_prev is not None:
                    self._btn_prev.setEnabled(False)
                # Re-enable forward-advance controls
                self._set_advance_enabled(True)
                self.statusBar().showMessage("Review Mode off — session resumed")
            else:
                try:
                    self._cursor.enter_review_mode()
                except RuntimeError as exc:
                    QtWidgets.QMessageBox.warning(self, "Review Mode", str(exc))
                    if self._review_act is not None:
                        self._review_act.setChecked(False)
                    return
                if self._review_banner is not None:
                    self._review_banner.setVisible(True)
                if self._review_act is not None:
                    self._review_act.setChecked(True)
                if self._btn_prev is not None:
                    self._btn_prev.setEnabled(True)
                # Disable forward-advance controls while reviewing
                self._set_advance_enabled(False)
                self.statusBar().showMessage(
                    "REVIEW MODE — press View > Review Mode again to resume"
                )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Review Mode Error", str(exc))

    def _step_back(self) -> None:
        if self._cursor is None or not self._cursor.in_review_mode:
            return
        try:
            self._cursor.step_back()
            if self._chart:
                self._chart.render_visible()
            self.refresh_status()
            self._update_order_panel_price_defaults()
        except RuntimeError:
            self.statusBar().showMessage("Already at the first bar.")

    # ------------------------------------------------------------------
    # Sim integration
    # ------------------------------------------------------------------

    def after_bar_advance(self) -> None:
        """Called after every bar advance.

        Runs the fill engine against the current bar, stores completed
        trades, then updates all status displays.
        """
        self._process_fills()
        self.refresh_status()
        self._update_order_panel_price_defaults()

        # Enable review mode as soon as there is at least one bar to step back to
        if self._cursor is not None and self._cursor.current_index > 0:
            if self._review_act is not None:
                self._review_act.setEnabled(True)

        # Update and save manifest on session completion
        if self._cursor is not None and self._cursor.is_complete:
            if (
                self._manifest is not None
                and self._trade_store is not None
                and self._manifest.status != "complete"
            ):
                from backtester.store.logs import compute_equity_curve
                trades    = self._trade_store.get_trades_for_session(self._session_id)
                curve     = compute_equity_curve(trades)
                final_eq  = float(curve.iloc[-1]) if len(curve) > 0 else 0.0
                self._manifest.update_completion(len(trades), final_eq)
                manifest_path = str(_PROJECT_DIR / f"{self._session_id}.manifest.json")
                self._manifest.save(manifest_path)

    def _process_fills(self) -> None:
        """Run FillEngine for the most recently revealed bar."""
        if self._fill_engine is None or self._cursor is None:
            return
        bar = self._cursor.visible_bars.iloc[-1]

        # Snapshot which positions are already open before this bar
        before_pos_ids = {p.order_id for p in self._fill_engine.open_positions}

        trades = self._fill_engine.process_bar(bar, self._pending_orders)

        # Draw entry markers for positions that just opened on this bar
        if self._chart is not None:
            for pos in self._fill_engine.open_positions:
                if pos.order_id not in before_pos_ids:
                    self._chart.add_entry_marker(
                        pos.entry_datetime, pos.entry_price, pos.side
                    )

        new_trades = False
        for trade in trades:
            if self._trade_store is not None:
                self._trade_store.insert_trade(trade)   # sets trade.db_id
                new_trades = True
            if self._chart is not None:
                self._chart.add_trade_marker(trade)

        # Remove filled / cancelled orders from pending list
        self._pending_orders = [o for o in self._pending_orders if o.status == "pending"]

        # Refresh chart order lines
        if self._chart is not None:
            self._chart.refresh_order_lines(
                self._pending_orders,
                self._fill_engine.open_positions,
            )

        # Refresh trade log whenever a new trade was completed
        if new_trades and self._order_panel is not None:
            self._order_panel.refresh_trade_log()

        if new_trades and self._summary_act is not None:
            self._summary_act.setEnabled(True)

    def _update_order_panel_price_defaults(self) -> None:
        """Pass current close and ATR14 to the order panel price inputs."""
        if self._order_panel is None or self._chart is None:
            return
        summary = self._chart.current_bar_summary()
        close   = float(summary["bar"]["close"])
        atr_val = summary["indicators"].get("ATR14", 0.0)
        if pd.isna(atr_val):
            atr_val = 0.0
        self._order_panel.update_price_defaults(close, float(atr_val))

    def _on_policy_changed(self, policy: str) -> None:
        """Update FillEngine when the user changes the ambiguity policy."""
        if self._fill_engine is not None:
            self._fill_engine.set_execution_rule(policy)

    def _on_order_placed(self, order: Order) -> None:
        """Receive a new order from :class:`OrderPanel`."""
        self._pending_orders.append(order)
        if self._chart is not None:
            self._chart.refresh_order_lines(
                self._pending_orders,
                self._fill_engine.open_positions if self._fill_engine else [],
            )

    def _show_summary(self) -> None:
        """Open the session summary dialog."""
        if self._trade_store is None:
            return
        trades = self._trade_store.get_trades_for_session(self._session_id)
        if not trades:
            QtWidgets.QMessageBox.information(
                self, "No Trades", "No trades in this session yet."
            )
            return
        try:
            from backtester.ui.summary_window import SummaryWindow
            dlg = SummaryWindow(self._session_id, self._trade_store, parent=self)
            dlg.exec()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Summary Error", f"Failed to open summary:\n{exc}"
            )

    # ------------------------------------------------------------------
    # Status / OHLC refresh
    # ------------------------------------------------------------------

    def refresh_status(self) -> None:
        """Update the info bar OHLC labels and the bottom status bar."""
        if self._cursor is None or self._chart is None:
            return

        summary = self._chart.current_bar_summary()
        bar     = summary["bar"]
        prev_c  = summary["prev_close"]

        # OHLC labels
        for lbl, key in [
            (self._lbl_o, "open"),
            (self._lbl_h, "high"),
            (self._lbl_l, "low"),
            (self._lbl_c, "close"),
        ]:
            if lbl is not None:
                lbl.setText(f"{bar[key]:.2f}")

        # Δ change
        if self._lbl_delta is not None:
            delta  = bar["close"] - prev_c
            pct    = (delta / prev_c * 100) if prev_c != 0 else 0.0
            sign   = "+" if delta >= 0 else ""
            colour = _GREEN if delta >= 0 else _RED
            self._lbl_delta.setStyleSheet(
                f"color: {colour}; font-size: 12px; background: transparent;"
            )
            self._lbl_delta.setText(f"{sign}{delta:.2f}  ({sign}{pct:.2f}%)")

        # Bottom status bar
        idx   = self._cursor.current_index
        total = len(self._cursor.bars)
        dt    = bar.name
        dt_str = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        self.statusBar().showMessage(f"Bar {idx + 1} / {total}  │  {dt_str}")

    # ------------------------------------------------------------------
    # Dark theme
    # ------------------------------------------------------------------

    def _set_dark_theme(self) -> None:
        palette = QtGui.QPalette()
        base = QtGui.QColor(_BAR_BG)
        text = QtGui.QColor(_BAR_FG)
        palette.setColor(QtGui.QPalette.ColorRole.Window,          base)
        palette.setColor(QtGui.QPalette.ColorRole.WindowText,      text)
        palette.setColor(QtGui.QPalette.ColorRole.Base,            base)
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase,   QtGui.QColor("#252932"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     base)
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText,     text)
        palette.setColor(QtGui.QPalette.ColorRole.Text,            text)
        palette.setColor(QtGui.QPalette.ColorRole.Button,          base)
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText,      text)
        palette.setColor(QtGui.QPalette.ColorRole.Highlight,       QtGui.QColor("#2962ff"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
        self.setPalette(palette)
