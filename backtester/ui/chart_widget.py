"""ChartWidget — PyQtGraph candlestick chart, TradingView-style."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from backtester.engine.anonymization import Anonymizer
from backtester.engine.cursor import BarCursor
from backtester.engine.indicators import atr as compute_atr, bollinger_bands, rsi, sma
from backtester.engine.masking import mask_series

# ── colour palette ───────────────────────────────────────────────────────────
_GREEN         = QtGui.QColor("#26a69a")   # bull green
_RED           = QtGui.QColor("#ef5350")   # bear red
_BG            = "#131722"                 # chart background
_BB_HEX        = "#4fc3f7"                 # Bollinger Band line colour
_BB_FILL_RGBA  = (79, 195, 247, 12)        # BB fill  (very transparent)
_PRICE_LINE    = "#787b86"                 # current-price line colour


# ── DateAxisItem ─────────────────────────────────────────────────────────────

class DateAxisItem(pg.AxisItem):
    """x-axis that displays YYYY-MM-DD labels instead of integer bar indices."""

    def __init__(self, bar_dates, **kwargs) -> None:
        super().__init__(orientation="bottom", **kwargs)
        if hasattr(bar_dates, "to_pydatetime"):
            self._dates: list = list(bar_dates.to_pydatetime())
        else:
            self._dates = list(bar_dates)

    def tickValues(self, minVal: float, maxVal: float, size: float):
        minVal, maxVal = sorted((minVal, maxVal))
        n = len(self._dates)
        if n == 0:
            return []
        n_vis  = max(1.0, maxVal - minVal)
        step   = max(1, int(n_vis / 10))
        first  = int(np.ceil(minVal))
        if step > 1:
            first = int(np.ceil(first / step)) * step
        major  = [i for i in range(first, int(maxVal) + 1, step) if 0 <= i < n]
        if not major:
            major = [max(0, min(n - 1, int(round(minVal))))]
        return [(step, major)]

    def tickStrings(self, values, scale: float, spacing: float):  # type: ignore[override]
        out = []
        for v in values:
            idx = int(round(v))
            if 0 <= idx < len(self._dates):
                dt = self._dates[idx]
                out.append(dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10])
            else:
                out.append("")
        return out

    def set_dates(self, bar_dates) -> None:
        """Replace the stored date list (used when toggling anonymization)."""
        if hasattr(bar_dates, "to_pydatetime"):
            self._dates = list(bar_dates.to_pydatetime())
        else:
            self._dates = list(bar_dates)
        self.picture = None   # invalidate cached axis picture
        self.update()


# ── CandlestickItem ──────────────────────────────────────────────────────────

class CandlestickItem(pg.GraphicsObject):
    """Paints OHLC candles directly in screen-pixel coordinates.

    Captures the data→device transform in ``paint()``, resets to identity,
    then draws 1-px wicks and filled bodies — no thick data-unit pens, no
    diamond/lens artefacts.
    """

    def __init__(self) -> None:
        super().__init__()
        self._x      = np.empty(0, dtype=np.float64)
        self._opens  = np.empty(0, dtype=np.float64)
        self._highs  = np.empty(0, dtype=np.float64)
        self._lows   = np.empty(0, dtype=np.float64)
        self._closes = np.empty(0, dtype=np.float64)

    def set_data(
        self,
        x:      np.ndarray,
        opens:  np.ndarray,
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
    ) -> None:
        self._x, self._opens = x, opens
        self._highs, self._lows, self._closes = highs, lows, closes
        self.informViewBoundsChanged()
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        if len(self._x) == 0:
            return QtCore.QRectF()
        lo, hi = float(self._lows.min()), float(self._highs.max())
        x0, x1 = float(self._x[0]), float(self._x[-1])
        return QtCore.QRectF(x0 - 0.5, lo, x1 - x0 + 1.0, max(hi - lo, 1e-9))

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        n = len(self._x)
        if n == 0:
            return

        t   = painter.transform()
        m11 = t.m11()
        m22 = t.m22()
        tdx = t.dx()
        tdy = t.dy()
        if m11 == 0 or m22 == 0:
            return

        bar_slot_px = abs(m11)
        body_w = max(2.0, min(8.0, bar_slot_px * 0.6))
        body_w = min(body_w, max(2.0, bar_slot_px - 1.0))

        xs   = self._x      * m11 + tdx
        hi_y = self._highs  * m22 + tdy
        lo_y = self._lows   * m22 + tdy
        op_y = self._opens  * m22 + tdy
        cl_y = self._closes * m22 + tdy

        # viewport culling
        if widget is not None:
            vw  = widget.width()
            vis = (xs + bar_slot_px >= 0) & (xs - bar_slot_px <= vw)
        else:
            vis = np.ones(n, dtype=bool)

        painter.save()
        painter.setTransform(QtGui.QTransform())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        green_pen    = QtGui.QPen(_GREEN, 1)
        red_pen      = QtGui.QPen(_RED,   1)
        green_brush  = QtGui.QBrush(_GREEN)
        red_brush    = QtGui.QBrush(_RED)
        bull_mask    = self._closes >= self._opens

        for i in range(n):
            if not vis[i]:
                continue
            px  = xs[i]
            phy = hi_y[i]
            ply = lo_y[i]
            poy = op_y[i]
            pcy = cl_y[i]

            bull = bool(bull_mask[i])
            painter.setPen(green_pen   if bull else red_pen)
            painter.setBrush(green_brush if bull else red_brush)

            painter.drawLine(QtCore.QPointF(px, ply), QtCore.QPointF(px, phy))

            body_top = min(poy, pcy)
            body_h   = abs(pcy - poy)
            if body_h < 1.0:
                painter.drawLine(
                    QtCore.QPointF(px - body_w * 0.5, poy),
                    QtCore.QPointF(px + body_w * 0.5, poy),
                )
            else:
                painter.drawRect(QtCore.QRectF(px - body_w * 0.5, body_top, body_w, body_h))

        painter.restore()


# ── ChartWidget ───────────────────────────────────────────────────────────────

class ChartWidget(pg.GraphicsLayoutWidget):
    """TradingView-style chart widget.

    Layout: single PlotItem filling the whole widget.
    OHLC / indicator text is displayed in the MainWindow info bar (not here).
    """

    # Indicator name → line colour (only for lines shown on the chart)
    _INDICATOR_COLOURS: dict[str, str] = {
        "BB_upper":  _BB_HEX,
        "BB_middle": _BB_HEX,
        "BB_lower":  _BB_HEX,
        "SMA20":     "#ffeb3b",
    }

    def __init__(
        self,
        cursor: BarCursor,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setBackground(_BG)
        self._cursor = cursor

        # ── Precompute indicators ─────────────────────────────────────
        close = cursor.bars["close"]
        high  = cursor.bars["high"]
        low   = cursor.bars["low"]
        bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
        self._indicator_series: dict[str, pd.Series] = {
            "BB_upper":  bb_u,
            "BB_middle": bb_m,
            "BB_lower":  bb_l,
            "SMA20":     sma(close, 20),
            "RSI14":     rsi(close, 14),
            "ATR14":     compute_atr(high, low, close, 14),
        }
        self._indicators_visible = True
        self._x_labels_visible   = True
        self._y_labels_visible   = True
        self._auto_scroll        = False   # enabled after initial fit_to_visible()

        # Anonymization state
        self._anonymizer:    Anonymizer | None = None
        self._is_anonymized: bool              = False
        # Keep original dates so we can swap back when de-anonymizing
        _idx = cursor.bars.index
        self._original_dates: list = (
            list(_idx.to_pydatetime()) if hasattr(_idx, "to_pydatetime")
            else list(_idx)
        )

        # ── Plot (single row, no HUD row) ────────────────────────────
        self._date_axis = DateAxisItem(cursor.bars.index)
        self._plot: pg.PlotItem = self.addPlot(
            row=0, col=0,
            axisItems={"bottom": self._date_axis},
        )
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=False, y=True, alpha=0.06)
        self._plot.getAxis("bottom").setTextPen(pg.mkPen("#555d6b"))
        self._plot.getAxis("left").setStyle(showValues=False)   # hide left axis
        self._plot.getAxis("left").setWidth(0)
        self._plot.showAxis("right")
        self._plot.getAxis("right").setTextPen(pg.mkPen("#8a9ba8"))
        self._plot.getAxis("right").setWidth(65)
        self._plot.getAxis("right").setStyle(showValues=True)

        # ── Candlestick item ──────────────────────────────────────────
        self._candle_item = CandlestickItem()
        self._plot.addItem(self._candle_item)

        # ── Bollinger Band lines ──────────────────────────────────────
        bb_pen    = pg.mkPen(_BB_HEX, width=1)
        bb_mid_pen = pg.mkPen(_BB_HEX, width=1,
                               style=QtCore.Qt.PenStyle.DashLine)
        self._bb_upper_item  = self._plot.plot([], [], pen=bb_pen,     antialias=False)
        self._bb_lower_item  = self._plot.plot([], [], pen=bb_pen,     antialias=False)
        self._bb_middle_item = self._plot.plot([], [], pen=bb_mid_pen, antialias=False)

        # Fill between upper and lower bands
        fill_brush = QtGui.QBrush(QtGui.QColor(*_BB_FILL_RGBA))
        self._bb_fill = pg.FillBetweenItem(
            self._bb_upper_item,
            self._bb_lower_item,
            brush=fill_brush,
        )
        self._plot.addItem(self._bb_fill)

        # ── SMA20 line ────────────────────────────────────────────────
        self._sma_item = self._plot.plot(
            [], [],
            pen=pg.mkPen("#ffeb3b", width=1),
            antialias=False,
        )

        # ── Current-price horizontal line ────────────────────────────
        self._price_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(_PRICE_LINE, width=1,
                         style=QtCore.Qt.PenStyle.DashLine),
            label="{value:.2f}",
            labelOpts={
                "position": 1.0,          # right side
                "color": "#c8cdd4",
                "fill": QtGui.QBrush(QtGui.QColor("#1e222d")),
                "movable": False,
            },
        )
        self._plot.addItem(self._price_line)

        # ── RSI panel (row 1) ─────────────────────────────────────────
        self._rsi_plot: pg.PlotItem = self.addPlot(row=1, col=0)
        self._rsi_plot.setXLink(self._plot)
        self.ci.layout.setRowFixedHeight(1, 100)
        self._rsi_plot.setMenuEnabled(False)
        self._rsi_plot.showGrid(x=False, y=True, alpha=0.06)
        self._rsi_plot.getAxis("left").setStyle(showValues=False)
        self._rsi_plot.getAxis("left").setWidth(0)
        self._rsi_plot.showAxis("right")
        self._rsi_plot.getAxis("right").setTextPen(pg.mkPen("#8a9ba8"))
        self._rsi_plot.getAxis("right").setWidth(65)
        self._rsi_plot.getAxis("right").setStyle(showValues=True)
        self._rsi_plot.getAxis("bottom").setStyle(showValues=False)
        self._rsi_plot.setYRange(0, 100, padding=0)

        # RSI reference lines at 30 and 70
        for lvl, colour in [(30, "#ef5350"), (70, "#26a69a")]:
            self._rsi_plot.addItem(pg.InfiniteLine(
                pos=lvl, angle=0, movable=False,
                pen=pg.mkPen(colour, width=1,
                             style=QtCore.Qt.PenStyle.DashLine),
            ))

        # RSI data line
        self._rsi_item = self._rsi_plot.plot(
            [], [],
            pen=pg.mkPen("#b39ddb", width=1),
            antialias=False,
        )

        # ── Initial render ────────────────────────────────────────────
        self.render_visible()       # _auto_scroll is still False here
        self.fit_to_visible()
        self._auto_scroll = True    # enable for all subsequent bar advances

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_visible(self) -> None:
        """Redraw all chart elements up to ``cursor.current_index``."""
        idx     = self._cursor.current_index
        visible = self._cursor.visible_bars
        n       = len(visible)

        x      = np.arange(n, dtype=np.float64)
        opens  = visible["open"].to_numpy(dtype=np.float64)
        highs  = visible["high"].to_numpy(dtype=np.float64)
        lows   = visible["low"].to_numpy(dtype=np.float64)
        closes = visible["close"].to_numpy(dtype=np.float64)

        # Apply anonymization transform to OHLC arrays if active
        anon = self._is_anonymized and self._anonymizer is not None
        if anon:
            opens  = self._anonymizer.transform_prices_array(opens)
            highs  = self._anonymizer.transform_prices_array(highs)
            lows   = self._anonymizer.transform_prices_array(lows)
            closes = self._anonymizer.transform_prices_array(closes)

        self._candle_item.set_data(x, opens, highs, lows, closes)

        # BB bands
        for name, item in [
            ("BB_upper",  self._bb_upper_item),
            ("BB_middle", self._bb_middle_item),
            ("BB_lower",  self._bb_lower_item),
        ]:
            if self._indicators_visible:
                masked = mask_series(self._indicator_series[name], idx).to_numpy(dtype=np.float64)
                if anon:
                    masked = self._anonymizer.transform_prices_array(masked)
                valid  = ~np.isnan(masked)
                item.setData(x[valid], masked[valid])
                item.setVisible(True)
            else:
                item.setVisible(False)

        self._bb_fill.setVisible(self._indicators_visible)

        # SMA20
        if self._indicators_visible:
            masked = mask_series(self._indicator_series["SMA20"], idx).to_numpy(dtype=np.float64)
            if anon:
                masked = self._anonymizer.transform_prices_array(masked)
            valid  = ~np.isnan(masked)
            self._sma_item.setData(x[valid], masked[valid])
            self._sma_item.setVisible(True)
        else:
            self._sma_item.setVisible(False)

        # Current-price line (in display space)
        self._price_line.setValue(float(closes[-1]))

        # RSI panel
        rsi_masked = mask_series(self._indicator_series["RSI14"], idx).to_numpy(dtype=np.float64)
        rsi_valid  = ~np.isnan(rsi_masked)
        if self._indicators_visible:
            self._rsi_item.setData(x[rsi_valid], rsi_masked[rsi_valid])
            self._rsi_plot.setVisible(True)
        else:
            self._rsi_plot.setVisible(False)

        # Auto-scroll: keep latest bar visible at the right edge
        if self._auto_scroll:
            self._maybe_scroll_to_latest()

    def current_bar_summary(self) -> dict:
        """Return OHLC + indicator snapshot for the MainWindow info bar."""
        idx  = self._cursor.current_index
        bar  = self._cursor.visible_bars.iloc[-1]
        prev = self._cursor.visible_bars.iloc[-2] if idx > 0 else bar
        vals = {
            name: float(s.iloc[idx])
            for name, s in self._indicator_series.items()
        }
        return {
            "bar":        bar,
            "prev_close": float(prev["close"]),
            "indicators": vals,
        }

    def _to_display_price(self, price: float) -> float:
        """Apply anonymization transform to a scalar price (no-op when inactive)."""
        if self._is_anonymized and self._anonymizer is not None:
            return self._anonymizer.transform_price(price)
        return price

    def fit_to_visible(self) -> None:
        """Fit both axes exactly to all currently visible bars."""
        bars = self._cursor.visible_bars
        if bars.empty:
            return
        n    = len(bars)
        # Map to display space (anonymized or real)
        y_lo = self._to_display_price(float(bars["low"].min()))
        y_hi = self._to_display_price(float(bars["high"].max()))
        # a is always > 0 so y_lo < y_hi is preserved
        if y_lo > y_hi:
            y_lo, y_hi = y_hi, y_lo
        y_m  = max((y_hi - y_lo) * 0.02, 0.01)
        vb   = self._plot.getViewBox()
        vb.setXRange(-0.5, n - 0.5 + max(1.0, n * 0.05), padding=0)
        vb.setYRange(y_lo - y_m, y_hi + y_m, padding=0)

    def _maybe_scroll_to_latest(self) -> None:
        """Pan right when the latest bar reaches the right viewport edge.

        Keeps the latest bar at ≈85% of the view width so there is always a
        small right margin, matching TradingView's replay-mode behaviour.
        Also re-fits the y-axis to the bars now in view.
        """
        vb = self._plot.getViewBox()
        x_left, x_right = vb.viewRange()[0]
        view_w  = x_right - x_left
        cur_x   = float(self._cursor.current_index)

        # Do nothing if the latest bar is still comfortably inside the view
        if cur_x <= x_right - 0.5:
            return

        # New x range: latest bar at 85 % from left, 15 % right margin
        new_right = cur_x + max(1.0, view_w * 0.15)
        new_left  = new_right - view_w

        vb.setXRange(new_left, new_right, padding=0)

        # Re-fit y to the bars visible in the new x window (in display space)
        bars  = self._cursor.visible_bars
        i_lo  = max(0, int(np.ceil(new_left)))
        i_hi  = min(len(bars) - 1, int(np.floor(new_right)))
        if i_lo <= i_hi:
            sl   = bars.iloc[i_lo : i_hi + 1]
            y_lo = self._to_display_price(float(sl["low"].min()))
            y_hi = self._to_display_price(float(sl["high"].max()))
            if y_lo > y_hi:
                y_lo, y_hi = y_hi, y_lo
            y_m  = max((y_hi - y_lo) * 0.02, 0.01)
            vb.setYRange(y_lo - y_m, y_hi + y_m, padding=0)

    def toggle_anonymization(self, anonymizer: Anonymizer | None = None) -> None:
        """Enable or disable anonymization of price and date data.

        Parameters
        ----------
        anonymizer:
            An already-configured :class:`Anonymizer` instance.  When
            *None* a default ``Anonymizer(seed=42)`` is lazy-created and
            fitted to the full bar series at the current cursor position.
        """
        if anonymizer is not None:
            self._anonymizer = anonymizer

        if self._anonymizer is None:
            # Lazy-create and fit on first toggle
            self._anonymizer = Anonymizer(seed=42)
            self._anonymizer.anonymize_prices(
                self._cursor.bars,
                start_index=self._cursor.current_index,
            )
            self._anonymizer.anonymize_dates(self._cursor.bars)

        self._is_anonymized = not self._is_anonymized

        # Swap x-axis date labels between real and anonymized
        if self._is_anonymized:
            anon_dates = self._anonymizer.get_anonymized_dates(self._original_dates)
            self._date_axis.set_dates(anon_dates)
        else:
            self._date_axis.set_dates(self._original_dates)

        self.render_visible()

    def toggle_indicators(self) -> None:
        self._indicators_visible = not self._indicators_visible
        self._rsi_plot.setVisible(self._indicators_visible)
        self.render_visible()

    def toggle_x_labels(self) -> None:
        self._x_labels_visible = not self._x_labels_visible
        self._plot.getAxis("bottom").setStyle(showValues=self._x_labels_visible)

    def toggle_y_labels(self) -> None:
        self._y_labels_visible = not self._y_labels_visible
        self._plot.getAxis("right").setStyle(showValues=self._y_labels_visible)

    # ------------------------------------------------------------------
    # Order / trade annotations
    # ------------------------------------------------------------------

    def refresh_order_lines(self, pending_orders, open_positions) -> None:
        """Redraw horizontal lines for pending entry orders and open brackets.

        Existing order lines are cleared and rebuilt from the current lists so
        this method is safe to call on every bar advance.

        Line styles per order type
        --------------------------
        * Limit buy:  blue  (#60a5fa) dashed,     label "LMT {price}"
        * Limit sell: amber (#f59e0b) dashed,     label "LMT {price}"
        * Stop buy:   red   (#f87171) dash-dot,   label "STP {price}"
        * Stop sell:  green (#34d399) dash-dot,   label "STP {price}"
        * Bracket stop:   red   (#ef5350) dashed (pending) / solid (open)
        * Bracket target: teal  (#26a69a) dashed (pending) / solid (open)
        """
        # Remove previous lines
        for item in getattr(self, "_order_line_items", []):
            self._plot.removeItem(item)
        self._order_line_items: list = []

        for order in pending_orders:
            if order.status != "pending":
                continue

            # ── Entry-price line for limit / stop-entry orders ─────────
            if order.order_type in ("limit", "stop") and order.price is not None:
                if order.order_type == "limit":
                    colour = "#60a5fa" if order.side == "long" else "#f59e0b"
                    style  = QtCore.Qt.PenStyle.DashLine
                    prefix = "LMT"
                else:
                    colour = "#f87171" if order.side == "long" else "#34d399"
                    style  = QtCore.Qt.PenStyle.DashDotLine
                    prefix = "STP"

                disp = self._to_display_price(order.price)
                line = pg.InfiniteLine(
                    pos=disp, angle=0, movable=False,
                    pen=pg.mkPen(colour, width=1, style=style),
                    label=f"{prefix} {order.price:.2f}",
                    labelOpts={
                        "position": 1.0,
                        "color":    colour,
                        "fill":     pg.mkBrush("#1e222d"),
                    },
                )
                self._plot.addItem(line)
                self._order_line_items.append(line)

            # ── Bracket stop / target for any pending order ─────────────
            for price, colour in [
                (order.stop_price,   "#ef5350"),
                (order.target_price, "#26a69a"),
            ]:
                if price is not None:
                    disp = self._to_display_price(price)
                    line = pg.InfiniteLine(
                        pos=disp, angle=0, movable=False,
                        pen=pg.mkPen(colour, width=1,
                                     style=QtCore.Qt.PenStyle.DashLine),
                    )
                    self._plot.addItem(line)
                    self._order_line_items.append(line)

        # Open positions — solid lines at stop and target
        for pos in open_positions:
            for price, colour in [
                (pos.stop_price,   "#ef5350"),
                (pos.target_price, "#26a69a"),
            ]:
                if price is not None:
                    disp = self._to_display_price(price)
                    line = pg.InfiniteLine(
                        pos=disp, angle=0, movable=False,
                        pen=pg.mkPen(colour, width=1),
                    )
                    self._plot.addItem(line)
                    self._order_line_items.append(line)

    def add_entry_marker(self, entry_dt: str, entry_price: float, side: str) -> None:
        """Draw an entry arrow for a newly opened position.

        Long entry: upward white triangle (symbol ``"t1"``).
        Short entry: downward white triangle (symbol ``"t"``).
        Appended to ``_trade_markers`` so it persists for the session lifetime.
        """
        bars    = self._cursor.bars
        dt_strs = [str(d) for d in bars.index]
        idx = next((i for i, s in enumerate(dt_strs) if s.startswith(entry_dt[:19])), None)
        if idx is None:
            return
        disp    = self._to_display_price(entry_price)
        symbol  = "t1" if side == "long" else "t"
        colour  = "#ffffff"
        scatter = pg.ScatterPlotItem(
            x=[idx], y=[disp],
            symbol=symbol, size=12,
            pen=pg.mkPen(colour, width=1),
            brush=pg.mkBrush(colour),
        )
        self._plot.addItem(scatter)
        markers = getattr(self, "_trade_markers", [])
        markers.append(scatter)
        self._trade_markers = markers

    def add_trade_marker(self, trade) -> None:
        """Paint entry and exit arrow markers for a completed trade."""
        bars       = self._cursor.bars
        entry_dt   = trade.entry_datetime
        exit_dt    = trade.exit_datetime

        # Locate bar indices by matching datetime strings
        dt_strs = [str(d) for d in bars.index]
        entry_idx = next((i for i, s in enumerate(dt_strs) if s.startswith(entry_dt[:19])), None)
        exit_idx  = next((i for i, s in enumerate(dt_strs) if s.startswith(exit_dt[:19])),  None)

        markers = getattr(self, "_trade_markers", [])

        for idx, price, symbol, colour in [
            (entry_idx, trade.entry_price, "t1" if trade.side == "long" else "t",  "#ffffff"),
            (exit_idx,  trade.exit_price,  "t"  if trade.side == "long" else "t1", "#ffeb3b"),
        ]:
            if idx is None:
                continue
            disp = self._to_display_price(price)
            scatter = pg.ScatterPlotItem(
                x=[idx], y=[disp],
                symbol=symbol,
                size=12,
                pen=pg.mkPen(colour, width=1),
                brush=pg.mkBrush(colour),
            )
            self._plot.addItem(scatter)
            markers.append(scatter)

        self._trade_markers = markers
