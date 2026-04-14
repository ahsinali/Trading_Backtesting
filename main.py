"""Entry point for the manual bar-advance backtesting tool."""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from backtester.ui.main_window import MainWindow


def main() -> None:
    pg.setConfigOptions(
        antialias=False,        # keep rendering fast at high bar counts
        foreground="#cccccc",
        background="#131722",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Backtester")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
