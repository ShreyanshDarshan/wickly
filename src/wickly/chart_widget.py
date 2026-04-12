"""Core PyQt6 chart widget for interactive candlestick charts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QFontMetrics,
    QWheelEvent, QMouseEvent, QPaintEvent, QResizeEvent, QPixmap,
)
from PyQt6.QtWidgets import QWidget, QToolTip
from PyQt6 import sip

from wickly.styles import _get_style, _MAV_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qcolor(hex_or_name: str, alpha: float = 1.0) -> QColor:
    """Create a QColor from a hex string or named colour with optional alpha."""
    c = QColor(hex_or_name)
    if alpha < 1.0:
        c.setAlphaF(alpha)
    return c


def _format_number(value: float) -> str:
    """Compact number formatter (e.g. 1.23M, 456K)."""
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class CandlestickWidget(QWidget):
    """A self-contained, interactive candlestick chart widget.

    Features
    --------
    - Candlestick / OHLC-bar / line / hollow-candle rendering
    - Optional volume sub-chart
    - Moving average overlays
    - Additional line / scatter plots (``addplot``)
    - Mouse-wheel zoom, click-drag pan
    - Crosshair + tooltip with OHLC / volume readout
    """

    # Emitted when visible range changes  (start_idx, end_idx)
    rangeChanged = pyqtSignal(int, int)

    # Margins (px)
    MARGIN_LEFT   = 12
    MARGIN_RIGHT  = 80
    MARGIN_TOP    = 40
    MARGIN_BOTTOM = 40

    # Chart / volume height ratio
    VOLUME_RATIO = 0.20  # 20 % of usable height when volume is shown

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        dates: np.ndarray | pd.DatetimeIndex | None = None,
        opens: np.ndarray | None = None,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        closes: np.ndarray | None = None,
        volumes: np.ndarray | None = None,
        chart_type: str = "candle",
        style: dict[str, Any] | None = None,
        show_volume: bool = False,
        mav: int | tuple[int, ...] | None = None,
        title: str | None = None,
        ylabel: str = "Price",
        addplots: list[dict[str, Any]] | None = None,
    ):
        super().__init__(parent)

        # --- data ---------------------------------------------------------------
        n = 0 if opens is None else len(opens)
        self._dates   = dates
        self._opens   = opens  if opens  is not None else np.array([])
        self._highs   = highs  if highs  is not None else np.array([])
        self._lows    = lows   if lows   is not None else np.array([])
        self._closes  = closes if closes is not None else np.array([])
        self._volumes = volumes
        self._n       = n

        # --- config -------------------------------------------------------------
        self._chart_type  = chart_type.lower()
        self._style       = _get_style(None) if style is None else style
        self._show_volume = show_volume and (volumes is not None)
        self._title       = title
        self._ylabel      = ylabel
        self._addplots    = addplots or []

        # moving averages (normalise to tuple)
        if mav is None:
            self._mavs: tuple[int, ...] = ()
        elif isinstance(mav, int):
            self._mavs = (mav,)
        else:
            self._mavs = tuple(mav)

        # Pre-compute MA lines
        self._mav_data: list[np.ndarray] = []
        for period in self._mavs:
            ma = pd.Series(self._closes).rolling(period).mean().values
            self._mav_data.append(ma)

        # --- view state ---------------------------------------------------------
        self._view_start = 0
        self._view_end   = max(n - 1, 0)

        # interaction
        self._dragging     = False
        self._drag_last_x  = 0
        self._mouse_pos: QPointF | None = None

        # appearance
        self.setMouseTracking(True)
        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_update(self) -> None:
        """Call ``self.update()`` only if the underlying C++ object is alive."""
        if not sip.isdeleted(self):
            self.update()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_data(
        self,
        dates: pd.DatetimeIndex,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> None:
        """Replace chart data and repaint."""
        self._dates   = dates
        self._opens   = opens
        self._highs   = highs
        self._lows    = lows
        self._closes  = closes
        self._volumes = volumes
        self._n       = len(opens)
        self._view_start = 0
        self._view_end   = max(self._n - 1, 0)
        self._recompute_mavs()
        self._safe_update()

    def append_data(
        self,
        dates: pd.DatetimeIndex,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray | None = None,
        *,
        auto_scroll: bool = True,
    ) -> None:
        """Append one or more bars to the chart.

        This is the primary method for building animated / live charts.
        New data is concatenated to the existing arrays and the view is
        optionally scrolled so the latest bar is always visible.

        Parameters
        ----------
        dates, opens, highs, lows, closes, volumes
            Array-like data for the new bar(s).  Must all have the same
            length.
        auto_scroll : bool
            If ``True`` (default) the visible window slides so the newest
            bar is at the right edge.  Set to ``False`` to keep the
            current pan position.
        """
        new_n = len(opens)
        was_at_end = (self._view_end >= self._n - 2)  # near the right edge

        self._dates  = self._dates.append(dates) if self._dates is not None else dates
        self._opens  = np.concatenate([self._opens, np.asarray(opens, dtype=float)])
        self._highs  = np.concatenate([self._highs, np.asarray(highs, dtype=float)])
        self._lows   = np.concatenate([self._lows, np.asarray(lows, dtype=float)])
        self._closes = np.concatenate([self._closes, np.asarray(closes, dtype=float)])
        if volumes is not None and self._volumes is not None:
            self._volumes = np.concatenate([self._volumes, np.asarray(volumes, dtype=float)])
        elif volumes is not None:
            self._volumes = np.asarray(volumes, dtype=float)

        self._n = len(self._opens)
        self._recompute_mavs()

        if auto_scroll and was_at_end:
            view_span = self._view_end - self._view_start
            self._view_end = self._n - 1
            self._view_start = max(0, self._view_end - view_span)

        self._safe_update()

    def update_last(
        self,
        open_: float | None = None,
        high: float | None = None,
        low: float | None = None,
        close: float | None = None,
        volume: float | None = None,
    ) -> None:
        """Update the most recent bar in-place and repaint.

        Use this for live tick updates within an incomplete candle.
        Only the supplied values are overwritten; pass ``None`` to leave
        a field unchanged.

        Parameters
        ----------
        open_ : float or None
            New open price.
        high : float or None
            New high price.
        low : float or None
            New low price.
        close : float or None
            New close price.
        volume : float or None
            New volume.
        """
        if self._n == 0:
            return
        idx = self._n - 1
        if open_ is not None:
            self._opens[idx] = open_
        if high is not None:
            self._highs[idx] = high
        if low is not None:
            self._lows[idx] = low
        if close is not None:
            self._closes[idx] = close
        if volume is not None and self._volumes is not None:
            self._volumes[idx] = volume
        self._recompute_mavs()
        self._safe_update()

    # ------------------------------------------------------------------
    # Live addplot helpers
    # ------------------------------------------------------------------

    def update_addplot(
        self,
        index: int,
        data: np.ndarray | list | pd.Series,
    ) -> None:
        """Replace the data of an existing addplot and repaint.

        Parameters
        ----------
        index : int
            Zero-based index into the list of addplots (in the order they
            were passed to ``plot()`` / ``live_plot()``).
        data : array-like
            New data for the overlay.  For ``type='line'`` or ``'scatter'``
            this must be a 1-D array-like of the same length as the current
            OHLCV data.  For ``type='segments'`` pass a list of
            ``(start_index, values)`` tuples.
        """
        if index < 0 or index >= len(self._addplots):
            raise IndexError(
                f"addplot index {index} out of range "
                f"(chart has {len(self._addplots)} addplot(s))"
            )
        ap = self._addplots[index]
        if ap["type"] == "segments":
            if not isinstance(data, list):
                raise TypeError(
                    "For type='segments', data must be a list of "
                    "(start_index, values) tuples."
                )
            segments: list[tuple[int, np.ndarray]] = []
            for item in data:
                if not (isinstance(item, (tuple, list)) and len(item) == 2):
                    raise ValueError(
                        "Each segment must be a (start_index, values) tuple."
                    )
                start, vals = item
                segments.append((int(start), np.asarray(vals, dtype=float)))
            ap["data"] = segments
        else:
            if isinstance(data, pd.Series):
                ap["data"] = data.values.astype(float)
            else:
                ap["data"] = np.asarray(data, dtype=float)
        self._safe_update()

    def append_addplot_data(
        self,
        index: int,
        values: np.ndarray | list | pd.Series | float,
    ) -> None:
        """Append values to an existing line or scatter addplot.

        Call this alongside :meth:`append_data` to keep an overlay array
        in sync with the OHLCV data.  For bars where the overlay has no
        value, pass ``float('nan')``.

        Parameters
        ----------
        index : int
            Zero-based index into the list of addplots.
        values : float or array-like
            Value(s) to append.  A single float appends one point;
            an array-like appends multiple points.
        """
        if index < 0 or index >= len(self._addplots):
            raise IndexError(
                f"addplot index {index} out of range "
                f"(chart has {len(self._addplots)} addplot(s))"
            )
        ap = self._addplots[index]
        if ap["type"] == "segments":
            raise TypeError(
                "append_addplot_data does not support type='segments'. "
                "Use update_addplot() to replace the full segment list."
            )
        if isinstance(values, (int, float)):
            new = np.array([float(values)])
        elif isinstance(values, pd.Series):
            new = values.values.astype(float)
        else:
            new = np.asarray(values, dtype=float)
        ap["data"] = np.concatenate([ap["data"], new])
        self._safe_update()

    def update_addplot_last(
        self,
        index: int,
        value: float,
    ) -> None:
        """Update the last value of a line/scatter addplot in-place.

        Use this alongside :meth:`update_last` to keep an overlay in
        sync with live tick updates on the most recent bar.

        Parameters
        ----------
        index : int
            Zero-based index into the list of addplots.
        value : float
            New value for the last point.
        """
        if index < 0 or index >= len(self._addplots):
            raise IndexError(
                f"addplot index {index} out of range "
                f"(chart has {len(self._addplots)} addplot(s))"
            )
        ap = self._addplots[index]
        if ap["type"] == "segments":
            raise TypeError(
                "update_addplot_last does not support type='segments'. "
                "Use update_addplot() to replace the full segment list."
            )
        if len(ap["data"]) == 0:
            return
        ap["data"][-1] = float(value)
        self._safe_update()

    def _recompute_mavs(self) -> None:
        """Recalculate all moving average arrays from current close data."""
        self._mav_data.clear()
        for period in self._mavs:
            ma = pd.Series(self._closes).rolling(period).mean().values
            self._mav_data.append(ma)

    def reset_view(self) -> None:
        self._view_start = 0
        self._view_end   = max(self._n - 1, 0)
        self._safe_update()

    def save(self, path: str) -> None:
        """Render the widget to an image file."""
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor(self._style.get("bg_color", "#ffffff")))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint(painter)
        painter.end()
        pixmap.save(path)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _chart_rect(self) -> QRectF:
        """Return the rectangle for the main (OHLC) chart area."""
        w = self.width()
        h = self.height()
        usable_h = h - self.MARGIN_TOP - self.MARGIN_BOTTOM
        if self._show_volume:
            chart_h = usable_h * (1.0 - self.VOLUME_RATIO)
        else:
            chart_h = usable_h
        return QRectF(
            self.MARGIN_LEFT,
            self.MARGIN_TOP,
            w - self.MARGIN_LEFT - self.MARGIN_RIGHT,
            chart_h,
        )

    def _volume_rect(self) -> QRectF:
        """Return the rectangle for the volume sub-chart."""
        w = self.width()
        h = self.height()
        usable_h = h - self.MARGIN_TOP - self.MARGIN_BOTTOM
        chart_h = usable_h * (1.0 - self.VOLUME_RATIO)
        vol_h   = usable_h * self.VOLUME_RATIO
        return QRectF(
            self.MARGIN_LEFT,
            self.MARGIN_TOP + chart_h,
            w - self.MARGIN_LEFT - self.MARGIN_RIGHT,
            vol_h,
        )

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _visible_range(self) -> tuple[int, int]:
        s = max(0, self._view_start)
        e = min(self._n - 1, self._view_end)
        return s, e

    def _price_range(self, s: int, e: int) -> tuple[float, float]:
        lo = float(np.nanmin(self._lows[s : e + 1]))
        hi = float(np.nanmax(self._highs[s : e + 1]))
        # include addplot data in range
        for ap in self._addplots:
            if ap["type"] == "segments":
                for seg_start, seg_data in ap["data"]:
                    seg_end = seg_start + len(seg_data)
                    # clip to visible range
                    lo_idx = max(s, seg_start)
                    hi_idx = min(e + 1, seg_end)
                    if lo_idx >= hi_idx:
                        continue
                    seg = seg_data[lo_idx - seg_start : hi_idx - seg_start]
                    valid = seg[~np.isnan(seg)]
                    if len(valid):
                        lo = min(lo, float(np.nanmin(valid)))
                        hi = max(hi, float(np.nanmax(valid)))
            else:
                ap_data = ap["data"]
                seg = ap_data[s : e + 1]
                valid = seg[~np.isnan(seg)]
                if len(valid):
                    lo = min(lo, float(np.nanmin(valid)))
                    hi = max(hi, float(np.nanmax(valid)))
        # include MAV data
        for ma in self._mav_data:
            seg = ma[s : e + 1]
            valid = seg[~np.isnan(seg)]
            if len(valid):
                lo = min(lo, float(np.nanmin(valid)))
                hi = max(hi, float(np.nanmax(valid)))
        pad = (hi - lo) * 0.05 if hi != lo else 1.0
        return lo - pad, hi + pad

    def _volume_range(self, s: int, e: int) -> tuple[float, float]:
        if self._volumes is None:
            return 0.0, 1.0
        mx = float(np.nanmax(self._volumes[s : e + 1]))
        return 0.0, mx * 1.15

    def _x_for_index(self, idx: int, rect: QRectF, s: int, e: int) -> float:
        count = e - s + 1
        if count <= 1:
            return rect.x() + rect.width() / 2
        return rect.x() + (idx - s) / (count - 1) * rect.width()

    def _index_for_x(self, x: float, rect: QRectF, s: int, e: int) -> int:
        count = e - s + 1
        if count <= 1:
            return s
        ratio = (x - rect.x()) / rect.width()
        return int(round(s + ratio * (count - 1)))

    def _y_for_price(self, price: float, rect: QRectF, lo: float, hi: float) -> float:
        if hi == lo:
            return rect.y() + rect.height() / 2
        return rect.y() + (1.0 - (price - lo) / (hi - lo)) * rect.height()

    def _y_for_volume(self, vol: float, rect: QRectF, vlo: float, vhi: float) -> float:
        if vhi == vlo:
            return rect.y() + rect.height()
        return rect.y() + (1.0 - (vol - vlo) / (vhi - vlo)) * rect.height()

    # ------------------------------------------------------------------
    # Candle width
    # ------------------------------------------------------------------

    def _candle_width(self, rect: QRectF, count: int) -> float:
        if count <= 1:
            return rect.width() * 0.4
        spacing = rect.width() / count
        return max(1.0, spacing * 0.7)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # background
        bg = _qcolor(self._style.get("bg_color", "#ffffff"))
        painter.fillRect(self.rect(), bg)
        self._paint(painter)
        painter.end()

    def _paint(self, painter: QPainter) -> None:  # noqa: C901 — complex but single responsibility
        if self._n == 0:
            return

        s, e = self._visible_range()
        if s > e:
            return

        chart_rect = self._chart_rect()
        plo, phi = self._price_range(s, e)
        count = e - s + 1
        cw = self._candle_width(chart_rect, count)

        style = self._style
        alpha = style.get("alpha", 1.0)

        # --- grid ---------------------------------------------------------------
        self._draw_grid(painter, chart_rect, plo, phi, style)

        # --- candles / bars / line ----------------------------------------------
        if self._chart_type in ("candle", "candlestick"):
            self._draw_candles(painter, chart_rect, s, e, plo, phi, cw, style, alpha)
        elif self._chart_type in ("ohlc", "ohlc_bars", "bars"):
            self._draw_ohlc(painter, chart_rect, s, e, plo, phi, cw, style, alpha)
        elif self._chart_type == "hollow":
            self._draw_hollow_candles(painter, chart_rect, s, e, plo, phi, cw, style, alpha)
        elif self._chart_type == "line":
            self._draw_line(painter, chart_rect, s, e, plo, phi, style)
        else:
            self._draw_candles(painter, chart_rect, s, e, plo, phi, cw, style, alpha)

        # --- moving averages ----------------------------------------------------
        self._draw_mavs(painter, chart_rect, s, e, plo, phi)

        # --- addplots -----------------------------------------------------------
        self._draw_addplots(painter, chart_rect, s, e, plo, phi)

        # --- price axis (right) -------------------------------------------------
        self._draw_price_axis(painter, chart_rect, plo, phi, style)

        # --- volume -------------------------------------------------------------
        if self._show_volume:
            vol_rect = self._volume_rect()
            vlo, vhi = self._volume_range(s, e)
            self._draw_volume(painter, vol_rect, s, e, vlo, vhi, cw, style, alpha)
            self._draw_volume_axis(painter, vol_rect, vlo, vhi, style)

        # --- date axis ----------------------------------------------------------
        bottom_rect = self._volume_rect() if self._show_volume else chart_rect
        self._draw_date_axis(painter, bottom_rect, s, e, style)

        # --- title --------------------------------------------------------------
        if self._title:
            painter.setPen(QPen(_qcolor(style.get("text_color", "#333"))))
            font = QFont("Segoe UI", 11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QRectF(self.MARGIN_LEFT, 2, chart_rect.width(), self.MARGIN_TOP - 4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._title,
            )

        # --- crosshair ----------------------------------------------------------
        if self._mouse_pos is not None:
            self._draw_crosshair(painter, chart_rect, s, e, plo, phi, style)

        # --- legend -------------------------------------------------------------
        self._draw_legend(painter, chart_rect, style)

    # ---- legend ----
    def _draw_legend(self, p: QPainter, rect: QRectF, style: dict) -> None:
        """Draw a compact legend in the top-left of the chart area."""
        entries: list[tuple[QColor, str, Qt.PenStyle]] = []

        # Collect MAV entries
        mav_colors = style.get("mavcolors", _MAV_COLORS)
        for idx, period in enumerate(self._mavs):
            color_str = mav_colors[idx % len(mav_colors)]
            entries.append((_qcolor(color_str), f"MA {period}", Qt.PenStyle.SolidLine))

        # Collect addplot entries (only those with a ylabel)
        for ap in self._addplots:
            label = ap.get("ylabel")
            if not label:
                continue
            color_str = ap.get("color") or "#1f77b4"
            ap_alpha = ap.get("alpha", 1.0)
            color = _qcolor(color_str, ap_alpha)
            ls_str = ap.get("linestyle", "-")
            if ls_str in ("--", "dashed"):
                pen_style = Qt.PenStyle.DashLine
            elif ls_str in ("-.", "dashdot"):
                pen_style = Qt.PenStyle.DashDotLine
            elif ls_str in (":", "dotted"):
                pen_style = Qt.PenStyle.DotLine
            else:
                pen_style = Qt.PenStyle.SolidLine
            # scatter uses a filled circle swatch; line/segments use a line swatch
            entries.append((color, label, pen_style))

        if not entries:
            return

        font = QFont("Segoe UI", 8)
        p.setFont(font)
        fm = QFontMetrics(font)

        swatch_w = 18.0  # width of the colour swatch line
        gap = 6.0        # gap between swatch and text
        item_gap = 14.0  # gap between consecutive entries
        pad_x = 8.0      # horizontal padding inside box
        pad_y = 4.0      # vertical padding inside box
        row_h = float(fm.height()) + 2.0

        # Compute total box size
        total_w = pad_x * 2
        for i, (_, label, _) in enumerate(entries):
            total_w += swatch_w + gap + fm.horizontalAdvance(label)
            if i < len(entries) - 1:
                total_w += item_gap
        total_h = row_h + pad_y * 2

        # Position: top-left inside chart, below any title
        box_x = rect.x() + 4
        box_y = rect.y() + 4

        # Draw background
        bg_color = _qcolor(style.get("bg_color", "#ffffff"), 0.80)
        border_color = _qcolor(style.get("grid_color", "#e0e0e0"), 0.60)
        p.setPen(QPen(border_color, 1))
        p.setBrush(QBrush(bg_color))
        p.drawRoundedRect(QRectF(box_x, box_y, total_w, total_h), 4, 4)

        # Draw entries
        text_color = _qcolor(style.get("text_color", "#333"))
        cx = box_x + pad_x
        cy_mid = box_y + pad_y + row_h / 2.0

        for color, label, pen_style in entries:
            # swatch
            pen = QPen(color, 2.0)
            pen.setStyle(pen_style)
            p.setPen(pen)
            p.drawLine(
                QPointF(cx, cy_mid),
                QPointF(cx + swatch_w, cy_mid),
            )
            cx += swatch_w + gap

            # text
            p.setPen(QPen(text_color))
            p.drawText(QPointF(cx, cy_mid + fm.ascent() / 2.0 - 1), label)
            cx += fm.horizontalAdvance(label) + item_gap

    # ---- grid ----
    def _draw_grid(self, p: QPainter, rect: QRectF, lo: float, hi: float, style: dict) -> None:
        pen = QPen(_qcolor(style.get("grid_color", "#e0e0e0")), 1, Qt.PenStyle.DotLine)
        p.setPen(pen)
        n_lines = 6
        for i in range(n_lines + 1):
            y = rect.y() + i * rect.height() / n_lines
            p.drawLine(QPointF(rect.x(), y), QPointF(rect.x() + rect.width(), y))

    # ---- candles ----
    def _draw_candles(self, p: QPainter, rect: QRectF, s: int, e: int,
                      lo: float, hi: float, cw: float, style: dict, alpha: float) -> None:
        for i in range(s, e + 1):
            x = self._x_for_index(i, rect, s, e)
            o, h, l, c = self._opens[i], self._highs[i], self._lows[i], self._closes[i]
            is_up = c >= o

            body_color = style["up_color"] if is_up else style["down_color"]
            edge_color = style.get("edge_up" if is_up else "edge_down", body_color)
            wick_color = style.get("wick_up" if is_up else "wick_down", edge_color)

            # wick
            p.setPen(QPen(_qcolor(wick_color, alpha), max(1.0, cw * 0.12)))
            wick_top = self._y_for_price(h, rect, lo, hi)
            wick_bot = self._y_for_price(l, rect, lo, hi)
            p.drawLine(QPointF(x, wick_top), QPointF(x, wick_bot))

            # body
            y_open  = self._y_for_price(o, rect, lo, hi)
            y_close = self._y_for_price(c, rect, lo, hi)
            body_top = min(y_open, y_close)
            body_h   = max(abs(y_open - y_close), 1.0)

            p.setPen(QPen(_qcolor(edge_color, alpha), 1))
            p.setBrush(QBrush(_qcolor(body_color, alpha)))
            p.drawRect(QRectF(x - cw / 2, body_top, cw, body_h))

    # ---- hollow candles ----
    def _draw_hollow_candles(self, p: QPainter, rect: QRectF, s: int, e: int,
                             lo: float, hi: float, cw: float, style: dict, alpha: float) -> None:
        for i in range(s, e + 1):
            x = self._x_for_index(i, rect, s, e)
            o, h, l, c = self._opens[i], self._highs[i], self._lows[i], self._closes[i]
            is_up = c >= o
            edge_color = style.get("edge_up" if is_up else "edge_down",
                                   style["up_color"] if is_up else style["down_color"])
            wick_color = style.get("wick_up" if is_up else "wick_down", edge_color)

            # wick
            p.setPen(QPen(_qcolor(wick_color, alpha), max(1.0, cw * 0.12)))
            p.drawLine(QPointF(x, self._y_for_price(h, rect, lo, hi)),
                       QPointF(x, self._y_for_price(l, rect, lo, hi)))

            y_open  = self._y_for_price(o, rect, lo, hi)
            y_close = self._y_for_price(c, rect, lo, hi)
            body_top = min(y_open, y_close)
            body_h   = max(abs(y_open - y_close), 1.0)
            p.setPen(QPen(_qcolor(edge_color, alpha), 1.5))
            if is_up:
                p.setBrush(QBrush(_qcolor(style.get("bg_color", "#ffffff"))))
            else:
                p.setBrush(QBrush(_qcolor(style["down_color"], alpha)))
            p.drawRect(QRectF(x - cw / 2, body_top, cw, body_h))

    # ---- OHLC bars ----
    def _draw_ohlc(self, p: QPainter, rect: QRectF, s: int, e: int,
                   lo: float, hi: float, cw: float, style: dict, alpha: float) -> None:
        for i in range(s, e + 1):
            x = self._x_for_index(i, rect, s, e)
            o, h, l, c = self._opens[i], self._highs[i], self._lows[i], self._closes[i]
            is_up = c >= o
            color = style["up_color"] if is_up else style["down_color"]
            pen = QPen(_qcolor(color, alpha), max(1.0, cw * 0.15))
            p.setPen(pen)

            y_h = self._y_for_price(h, rect, lo, hi)
            y_l = self._y_for_price(l, rect, lo, hi)
            y_o = self._y_for_price(o, rect, lo, hi)
            y_c = self._y_for_price(c, rect, lo, hi)

            # vertical bar
            p.drawLine(QPointF(x, y_h), QPointF(x, y_l))
            # open tick (left)
            p.drawLine(QPointF(x - cw / 2, y_o), QPointF(x, y_o))
            # close tick (right)
            p.drawLine(QPointF(x, y_c), QPointF(x + cw / 2, y_c))

    # ---- line ----
    def _draw_line(self, p: QPainter, rect: QRectF, s: int, e: int,
                   lo: float, hi: float, style: dict) -> None:
        pen = QPen(_qcolor(style.get("up_color", "#26a69a")), 1.8)
        p.setPen(pen)
        points = []
        for i in range(s, e + 1):
            x = self._x_for_index(i, rect, s, e)
            y = self._y_for_price(self._closes[i], rect, lo, hi)
            points.append(QPointF(x, y))
        for j in range(len(points) - 1):
            p.drawLine(points[j], points[j + 1])

    # ---- moving averages ----
    def _draw_mavs(self, p: QPainter, rect: QRectF, s: int, e: int,
                   lo: float, hi: float) -> None:
        mav_colors = self._style.get("mavcolors", _MAV_COLORS)
        for idx, ma_data in enumerate(self._mav_data):
            color = mav_colors[idx % len(mav_colors)]
            pen = QPen(_qcolor(color), 1.4)
            pen.setStyle(Qt.PenStyle.SolidLine)
            p.setPen(pen)
            prev: QPointF | None = None
            for i in range(s, e + 1):
                val = ma_data[i]
                if np.isnan(val):
                    prev = None
                    continue
                pt = QPointF(
                    self._x_for_index(i, rect, s, e),
                    self._y_for_price(val, rect, lo, hi),
                )
                if prev is not None:
                    p.drawLine(prev, pt)
                prev = pt

    # ---- addplots ----
    def _draw_addplots(self, p: QPainter, rect: QRectF, s: int, e: int,
                       lo: float, hi: float) -> None:
        for ap in self._addplots:
            color_str = ap.get("color") or "#1f77b4"
            ap_alpha = ap.get("alpha", 1.0)
            color = _qcolor(color_str, ap_alpha)

            if ap["type"] == "segments":
                self._draw_segments(p, rect, s, e, lo, hi, ap, color)
            elif ap["type"] == "scatter":
                data = ap["data"]
                marker_size = ap.get("markersize", 50)
                radius = math.sqrt(marker_size) / 2
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(color))
                for i in range(s, e + 1):
                    val = data[i] if i < len(data) else float("nan")
                    if np.isnan(val):
                        continue
                    x = self._x_for_index(i, rect, s, e)
                    y = self._y_for_price(val, rect, lo, hi)
                    p.drawEllipse(QPointF(x, y), radius, radius)
            else:
                # line
                data = ap["data"]
                lw = ap.get("width", 1.5)
                pen = QPen(color, lw)
                ls = ap.get("linestyle", "-")
                if ls in ("--", "dashed"):
                    pen.setStyle(Qt.PenStyle.DashLine)
                elif ls in ("-.", "dashdot"):
                    pen.setStyle(Qt.PenStyle.DashDotLine)
                elif ls in (":", "dotted"):
                    pen.setStyle(Qt.PenStyle.DotLine)
                p.setPen(pen)
                prev: QPointF | None = None  # type: ignore[no-redef]
                for i in range(s, e + 1):
                    val = data[i] if i < len(data) else float("nan")
                    if np.isnan(val):
                        prev = None
                        continue
                    pt = QPointF(
                        self._x_for_index(i, rect, s, e),
                        self._y_for_price(val, rect, lo, hi),
                    )
                    if prev is not None:
                        p.drawLine(prev, pt)
                    prev = pt

    def _draw_segments(self, p: QPainter, rect: QRectF, s: int, e: int,
                       lo: float, hi: float, ap: dict, color: QColor) -> None:
        """Draw independent (possibly overlapping) line segments."""
        lw = ap.get("width", 1.5)
        pen = QPen(color, lw)
        ls = ap.get("linestyle", "-")
        if ls in ("--", "dashed"):
            pen.setStyle(Qt.PenStyle.DashLine)
        elif ls in ("-.", "dashdot"):
            pen.setStyle(Qt.PenStyle.DashDotLine)
        elif ls in (":", "dotted"):
            pen.setStyle(Qt.PenStyle.DotLine)
        p.setPen(pen)

        for seg_start, seg_data in ap["data"]:
            seg_end = seg_start + len(seg_data)
            # Clip to visible range
            lo_idx = max(s, seg_start)
            hi_idx = min(e, seg_end - 1)
            if lo_idx > hi_idx:
                continue
            prev: QPointF | None = None
            for i in range(lo_idx, hi_idx + 1):
                val = seg_data[i - seg_start]
                if np.isnan(val):
                    prev = None
                    continue
                pt = QPointF(
                    self._x_for_index(i, rect, s, e),
                    self._y_for_price(val, rect, lo, hi),
                )
                if prev is not None:
                    p.drawLine(prev, pt)
                prev = pt

    # ---- volume ----
    def _draw_volume(self, p: QPainter, rect: QRectF, s: int, e: int,
                     vlo: float, vhi: float, cw: float, style: dict, alpha: float) -> None:
        if self._volumes is None:
            return
        # light separator line
        p.setPen(QPen(_qcolor(style.get("grid_color", "#e0e0e0")), 1))
        p.drawLine(QPointF(rect.x(), rect.y()), QPointF(rect.x() + rect.width(), rect.y()))

        for i in range(s, e + 1):
            x = self._x_for_index(i, rect, s, e)
            vol = self._volumes[i]
            is_up = self._closes[i] >= self._opens[i]
            color = style.get("volume_up" if is_up else "volume_down",
                              style["up_color"] if is_up else style["down_color"])
            y_top = self._y_for_volume(vol, rect, vlo, vhi)
            y_bot = rect.y() + rect.height()
            bar_h = y_bot - y_top

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_qcolor(color, alpha * 0.7)))
            p.drawRect(QRectF(x - cw / 2, y_top, cw, bar_h))

    # ---- axes ----
    def _draw_price_axis(self, p: QPainter, rect: QRectF, lo: float, hi: float, style: dict) -> None:
        text_color = _qcolor(style.get("text_color", "#333"))
        p.setPen(QPen(text_color))
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        fm = QFontMetrics(font)
        n_ticks = 6
        x_label = rect.x() + rect.width() + 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            price = hi - frac * (hi - lo)
            y = rect.y() + frac * rect.height()
            label = f"{price:,.2f}"
            p.drawText(QPointF(x_label, y + fm.ascent() / 2), label)

    def _draw_volume_axis(self, p: QPainter, rect: QRectF, vlo: float, vhi: float, style: dict) -> None:
        text_color = _qcolor(style.get("text_color", "#333"))
        p.setPen(QPen(text_color))
        font = QFont("Segoe UI", 7)
        p.setFont(font)
        fm = QFontMetrics(font)
        x_label = rect.x() + rect.width() + 6
        for i in (0, 1):
            frac = i
            vol = vhi - frac * (vhi - vlo)
            y = rect.y() + frac * rect.height()
            label = _format_number(vol)
            p.drawText(QPointF(x_label, y + fm.ascent() / 2), label)
        # "Vol" label
        p.drawText(QPointF(rect.x() + 4, rect.y() + fm.ascent() + 2), "Vol")

    def _draw_date_axis(self, p: QPainter, rect: QRectF, s: int, e: int, style: dict) -> None:
        if self._dates is None:
            return
        text_color = _qcolor(style.get("text_color", "#333"))
        p.setPen(QPen(text_color))
        font = QFont("Segoe UI", 7)
        p.setFont(font)
        fm = QFontMetrics(font)
        count = e - s + 1
        # show at most ~8 labels
        step = max(1, count // 8)
        y_label = rect.y() + rect.height() + fm.ascent() + 4
        for i in range(s, e + 1, step):
            x = self._x_for_index(i, rect, s, e)
            dt = self._dates[i]
            if hasattr(dt, "strftime"):
                label = dt.strftime("%b %d")
            else:
                label = str(dt)
            tw = fm.horizontalAdvance(label)
            p.drawText(QPointF(x - tw / 2, y_label), label)

    # ---- crosshair ----
    def _draw_crosshair(self, p: QPainter, rect: QRectF, s: int, e: int,
                        lo: float, hi: float, style: dict) -> None:
        mx = self._mouse_pos.x()  # type: ignore[union-attr]
        my = self._mouse_pos.y()  # type: ignore[union-attr]

        # Only draw inside chart area (including volume)
        full_bottom = (self._volume_rect().y() + self._volume_rect().height()) if self._show_volume else (rect.y() + rect.height())
        if not (rect.x() <= mx <= rect.x() + rect.width() and rect.y() <= my <= full_bottom):
            return

        pen = QPen(_qcolor(style.get("text_color", "#333"), 0.4), 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(rect.x(), my), QPointF(rect.x() + rect.width(), my))
        p.drawLine(QPointF(mx, rect.y()), QPointF(mx, full_bottom))

        # index under cursor
        idx = self._index_for_x(mx, rect, s, e)
        idx = max(s, min(e, idx))

        # tooltip
        dt_str = ""
        if self._dates is not None and hasattr(self._dates[idx], "strftime"):
            dt_str = self._dates[idx].strftime("%Y-%m-%d")
        parts = [
            dt_str,
            f"O {self._opens[idx]:,.2f}",
            f"H {self._highs[idx]:,.2f}",
            f"L {self._lows[idx]:,.2f}",
            f"C {self._closes[idx]:,.2f}",
        ]
        if self._volumes is not None:
            parts.append(f"V {_format_number(self._volumes[idx])}")
        tip = "  |  ".join(parts)

        # draw label background
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(tip) + 12
        th = fm.height() + 6

        label_x = min(mx + 14, rect.x() + rect.width() - tw - 4)
        label_y = max(rect.y(), my - th - 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(_qcolor("#222222", 0.85)))
        p.drawRoundedRect(QRectF(label_x, label_y, tw, th), 4, 4)
        p.setPen(QPen(QColor("#ffffff")))
        p.drawText(QPointF(label_x + 6, label_y + fm.ascent() + 3), tip)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        count = self._view_end - self._view_start + 1
        zoom_amount = max(1, int(count * 0.1))

        if delta > 0:
            # zoom in
            self._view_start = min(self._view_start + zoom_amount, self._view_end - 2)
            self._view_end   = max(self._view_end - zoom_amount, self._view_start + 2)
        else:
            # zoom out
            self._view_start = max(0, self._view_start - zoom_amount)
            self._view_end   = min(self._n - 1, self._view_end + zoom_amount)

        self.rangeChanged.emit(self._view_start, self._view_end)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_last_x = int(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        self._mouse_pos = pos

        if self._dragging:
            dx = int(pos.x()) - self._drag_last_x
            self._drag_last_x = int(pos.x())
            rect = self._chart_rect()
            count = self._view_end - self._view_start + 1
            idx_shift = int(-dx / (rect.width() / max(count, 1)))
            new_s = self._view_start + idx_shift
            new_e = self._view_end + idx_shift
            if 0 <= new_s and new_e <= self._n - 1:
                self._view_start = new_s
                self._view_end = new_e
                self.rangeChanged.emit(self._view_start, self._view_end)

        self.update()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._mouse_pos = None
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        self.update()
