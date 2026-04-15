"""IndicatorsDialog — lets the user choose and configure the overlay indicator."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal

from backtester.engine.indicators import IndicatorConfig

# ── style constants (match app dark palette) ────────────────────────────────
_BG      = "#1e222d"
_FG      = "#c8cdd4"
_HDR_BG  = "#252932"
_DIM     = "#555d6b"
_INPUT_BG = "#131722"
_BORDER  = "#2a2e39"

_DIALOG_CSS = f"""
    QDialog, QWidget {{
        background: {_BG};
        color: {_FG};
    }}
    QLabel {{
        color: {_FG};
        font-size: 12px;
    }}
    QComboBox {{
        background: {_INPUT_BG};
        color: {_FG};
        border: 1px solid {_BORDER};
        border-radius: 3px;
        padding: 4px 8px;
        font-size: 12px;
        min-height: 24px;
    }}
    QComboBox QAbstractItemView {{
        background: {_INPUT_BG};
        color: {_FG};
        selection-background-color: #1e3a5f;
    }}
    QSpinBox, QDoubleSpinBox {{
        background: {_INPUT_BG};
        color: {_FG};
        border: 1px solid {_BORDER};
        border-radius: 3px;
        padding: 3px 6px;
        font-size: 12px;
        min-height: 22px;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background: {_HDR_BG};
        border: none;
    }}
    QPushButton {{
        background: {_HDR_BG};
        color: {_FG};
        border: 1px solid {_BORDER};
        border-radius: 3px;
        padding: 5px 18px;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background: #363a45;
    }}
    QPushButton:pressed {{
        background: #1e3a5f;
    }}
    QGroupBox {{
        color: {_DIM};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 6px;
        font-size: 11px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
    }}
"""

_HINT_CSS = f"color: {_DIM}; font-size: 11px;"


def _lbl(text: str, hint: bool = False) -> QtWidgets.QLabel:
    l = QtWidgets.QLabel(text)
    if hint:
        l.setStyleSheet(_HINT_CSS)
    return l


class IndicatorsDialog(QtWidgets.QDialog):
    """Configuration dialog for the chart overlay indicator.

    Emits :attr:`config_changed` with an :class:`~backtester.engine.indicators.IndicatorConfig`
    when the user clicks Apply.

    Parameters
    ----------
    current_config:
        The currently active indicator configuration (pre-populates the fields).
    parent:
        Optional parent widget.
    """

    config_changed: Signal = Signal(object)   # emits IndicatorConfig

    def __init__(
        self,
        current_config: IndicatorConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        cfg = current_config or IndicatorConfig()

        self.setWindowTitle("Indicators")
        self.setFixedSize(360, 320)
        self.setStyleSheet(_DIALOG_CSS)

        # ── Mode selector ──────────────────────────────────────────────
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(["Simple Moving Average", "Keltner Channel"])
        self._combo.setCurrentIndex(0 if cfg.mode == "sma" else 1)

        # ── Stacked pages ─────────────────────────────────────────────
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._build_sma_page(cfg))
        self._stack.addWidget(self._build_keltner_page(cfg))
        self._stack.setCurrentIndex(0 if cfg.mode == "sma" else 1)

        self._combo.currentIndexChanged.connect(self._stack.setCurrentIndex)

        # ── Buttons ───────────────────────────────────────────────────
        btn_apply  = QtWidgets.QPushButton("Apply")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_apply.clicked.connect(self._apply)
        btn_cancel.clicked.connect(self.reject)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_cancel)

        # ── Main layout ────────────────────────────────────────────────
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(_lbl("Overlay type:"))
        mode_row.addWidget(self._combo, 1)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(16, 16, 16, 16)
        vbox.setSpacing(12)
        vbox.addLayout(mode_row)
        vbox.addWidget(self._stack, 1)
        vbox.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _build_sma_page(self, cfg: IndicatorConfig) -> QtWidgets.QWidget:
        """Page 0 — Simple Moving Average settings."""
        self._sma_period  = QtWidgets.QSpinBox()
        self._sma_period.setRange(2, 500)
        self._sma_period.setValue(cfg.sma_period)

        self._sma_period2 = QtWidgets.QSpinBox()
        self._sma_period2.setRange(2, 500)
        self._sma_period2.setValue(cfg.sma_period_2)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.addRow("Period 1:", self._sma_period)
        form.addRow("", _lbl("SMA(period 1) — blue line", hint=True))
        form.addRow("Period 2:", self._sma_period2)
        form.addRow("", _lbl("SMA(period 2) — yellow line", hint=True))

        w = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(form)
        vbox.addStretch()
        return w

    def _build_keltner_page(self, cfg: IndicatorConfig) -> QtWidgets.QWidget:
        """Page 1 — Keltner Channel settings."""
        self._kc_ema_period = QtWidgets.QSpinBox()
        self._kc_ema_period.setRange(2, 500)
        self._kc_ema_period.setValue(cfg.keltner_ema_period)

        self._kc_atr_period = QtWidgets.QSpinBox()
        self._kc_atr_period.setRange(2, 500)
        self._kc_atr_period.setValue(cfg.keltner_atr_period)

        self._kc_multiplier = QtWidgets.QDoubleSpinBox()
        self._kc_multiplier.setRange(0.1, 10.0)
        self._kc_multiplier.setSingleStep(0.1)
        self._kc_multiplier.setDecimals(1)
        self._kc_multiplier.setValue(cfg.keltner_atr_multiplier)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.addRow("EMA Period:",     self._kc_ema_period)
        form.addRow("ATR Period:",     self._kc_atr_period)
        form.addRow("ATR Multiplier:", self._kc_multiplier)
        form.addRow("", _lbl("Upper = EMA + (mult × ATR)", hint=True))
        form.addRow("", _lbl("Lower = EMA − (mult × ATR)", hint=True))

        w = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(form)
        vbox.addStretch()
        return w

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _apply(self) -> None:
        mode = "sma" if self._combo.currentIndex() == 0 else "keltner"
        cfg = IndicatorConfig(
            mode                   = mode,
            sma_period             = self._sma_period.value(),
            sma_period_2           = self._sma_period2.value(),
            keltner_ema_period     = self._kc_ema_period.value(),
            keltner_atr_period     = self._kc_atr_period.value(),
            keltner_atr_multiplier = self._kc_multiplier.value(),
        )
        self.config_changed.emit(cfg)
        self.accept()
