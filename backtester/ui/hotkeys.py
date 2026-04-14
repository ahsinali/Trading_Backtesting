"""Hotkey installation for the backtester main window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

# Use a module-level list so previous shortcuts can be removed when
# install_hotkeys is called again (e.g., after opening a second file).
_active_shortcuts: list[QShortcut] = []


def install_hotkeys(window, cursor, chart) -> None:  # noqa: ANN001
    """Bind keyboard shortcuts to backtester actions.

    Parameters
    ----------
    window:
        The ``MainWindow`` instance.  Must expose ``refresh_status()``.
    cursor:
        Active ``BarCursor``.
    chart:
        Active ``ChartWidget``.
    """
    # Remove any shortcuts installed by a previous call
    for sc in _active_shortcuts:
        sc.setEnabled(False)
        sc.deleteLater()
    _active_shortcuts.clear()

    def _bind(key: str | Qt.Key, callback) -> None:
        sc = QShortcut(QKeySequence(key), window)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(callback)
        _active_shortcuts.append(sc)

    # ── Space — advance one bar (blocked while in review mode) ───────
    def _advance() -> None:
        if getattr(window, "_advance_paused", False):
            return
        try:
            cursor.advance()
            chart.render_visible()
            window.after_bar_advance()
        except StopIteration:
            window.statusBar().showMessage(
                f"End of data — {len(cursor.bars)} bars total."
            )

    _bind(Qt.Key.Key_Space, _advance)

    # ── F — fit chart to visible data ────────────────────────────────
    _bind(Qt.Key.Key_F, chart.fit_to_visible)

    # ── A — toggle anonymization ─────────────────────────────────────
    _bind(Qt.Key.Key_A, chart.toggle_anonymization)

    # ── I — toggle indicators ────────────────────────────────────────
    _bind(Qt.Key.Key_I, chart.toggle_indicators)

    # ── X — toggle x-axis labels ─────────────────────────────────────
    _bind(Qt.Key.Key_X, chart.toggle_x_labels)

    # ── Y — toggle y-axis labels ─────────────────────────────────────
    _bind(Qt.Key.Key_Y, chart.toggle_y_labels)

    # ── Left Arrow — step back one bar (review mode only) ────────────
    def _step_back() -> None:
        if cursor.in_review_mode:
            try:
                cursor.step_back()
                chart.render_visible()
                window.refresh_status()
                window._update_order_panel_price_defaults()
            except RuntimeError:
                window.statusBar().showMessage("Already at the first bar.")

    _bind(Qt.Key.Key_Left, _step_back)
