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
    QPolygonF,
)
from PyQt6.QtWidgets import QWidget, QToolTip
from PyQt6 import sip

from wickly.addplot import SubPanel
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

    # Level-of-detail budget: if the visible range exceeds this many bars
    # the data is aggregated to a lower temporal resolution for rendering.
    CANDLE_BUDGET = 500

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
        panels: list[SubPanel] | None = None,
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
        self._sub_panels: list[SubPanel] = list(panels) if panels else []

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
        self._pan_offset = 0.0   # fractional bar offset for smooth panning
        self._agg_orig_view: tuple[int, int] | None = None

        # interaction
        self._dragging     = False
        self._drag_last_x  = 0.0
        self._mouse_pos: QPointF | None = None

        # legend visibility — one bool per series
        self._mav_visible:     list[bool] = [True] * len(self._mavs)
        self._addplot_visible: list[bool] = [True] * len(self._addplots)
        self._volume_visible:  bool = True

        # legend interaction state (rebuilt each paint)
        self._legend_hover_idx:  int | None = None
        self._legend_hit_areas:  list[tuple[QRectF, QRectF, str, int]] = []
        self._legend_rect:       QRectF | None = None

        # panel resize drag state
        self._resizing_sep: int | None = None  # index into _separator_kinds
        self._resize_last_y: float = 0.0
        self._separator_kinds: list[tuple[str, int]] = []
        # Each entry: ("vol_top", -1), ("panel_top", panel_idx), etc.

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
        self._pan_offset = 0.0
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
            self._pan_offset = 0.0

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
        self._pan_offset = 0.0
        self._safe_update()

    def set_series_visible(self, kind: str, index: int, visible: bool) -> None:
        """Programmatically toggle the visibility of an overlay series.

        Parameters
        ----------
        kind : {'mav', 'addplot', 'volume'}
            Which series to address.
        index : int
            Zero-based index within that series list (ignored for ``'volume'``).
        visible : bool
            ``True`` to show, ``False`` to hide.
        """
        if kind == "mav":
            self._mav_visible[index] = visible
        elif kind == "addplot":
            self._addplot_visible[index] = visible
        elif kind == "volume":
            self._volume_visible = visible
        elif kind == "panel":
            self._sub_panels[index].visible = visible
        else:
            raise ValueError(f"Unknown kind {kind!r}. Use 'mav', 'addplot', 'volume', or 'panel'.")
        self._safe_update()

    # ------------------------------------------------------------------
    # Sub-panel public API
    # ------------------------------------------------------------------

    def add_panel(self, panel: SubPanel) -> int:
        """Append a sub-panel and return its index."""
        self._sub_panels.append(panel)
        self._safe_update()
        return len(self._sub_panels) - 1

    def remove_panel(self, index: int) -> None:
        """Remove the sub-panel at *index*."""
        del self._sub_panels[index]
        self._safe_update()

    def set_panel_data(self, index: int, data: np.ndarray) -> None:
        """Replace the data array of sub-panel *index* and repaint."""
        self._sub_panels[index].data = np.asarray(data, dtype=float)
        self._safe_update()

    def append_panel_data(self, index: int, value: float) -> None:
        """Append one scalar value to sub-panel *index* and repaint."""
        self._sub_panels[index].data = np.append(self._sub_panels[index].data, float(value))
        self._safe_update()

    def update_panel_last(self, index: int, value: float) -> None:
        """Update the last element of sub-panel *index* in-place and repaint."""
        self._sub_panels[index].data[-1] = float(value)
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

    def _layout_panels(self) -> tuple[QRectF, QRectF | None, list[QRectF]]:
        """Compute rects for the main chart, volume sub-chart, and user sub-panels.

        Returns
        -------
        main_rect : QRectF
            Rectangle for the main OHLCV chart area.
        vol_rect : QRectF or None
            Rectangle for the volume sub-chart, or ``None`` if not shown.
        panel_rects : list[QRectF]
            One rect per entry in ``_sub_panels`` (zero-height when hidden).
        """
        w = self.width()
        h = self.height()
        usable_h = h - self.MARGIN_TOP - self.MARGIN_BOTTOM
        x  = float(self.MARGIN_LEFT)
        pw = w - self.MARGIN_LEFT - self.MARGIN_RIGHT

        # Effective height ratios (zero when hidden)
        vol_ratio = self.VOLUME_RATIO if (self._show_volume and self._volume_visible) else 0.0
        panel_ratios = [
            p.height_ratio if p.visible else 0.0
            for p in self._sub_panels
        ]
        total_sub = vol_ratio + sum(panel_ratios)

        # Clamp so main chart always gets at least 25 %
        if total_sub > 0.75:
            scale = 0.75 / total_sub
            vol_ratio *= scale
            panel_ratios = [r * scale for r in panel_ratios]
            total_sub = 0.75

        main_h = usable_h * (1.0 - total_sub)
        main_rect = QRectF(x, self.MARGIN_TOP, pw, main_h)

        # Volume
        y = self.MARGIN_TOP + main_h
        if self._show_volume and self._volume_visible:
            vol_h   = usable_h * vol_ratio
            vol_rect: QRectF | None = QRectF(x, y, pw, vol_h)
            y += vol_h
        else:
            vol_rect = None

        # User sub-panels
        panel_rects: list[QRectF] = []
        for i, panel in enumerate(self._sub_panels):
            if panel.visible:
                ph = usable_h * panel_ratios[i]
                panel_rects.append(QRectF(x, y, pw, ph))
                y += ph
            else:
                panel_rects.append(QRectF(x, y, pw, 0.0))

        return main_rect, vol_rect, panel_rects

    _SEPARATOR_GRAB = 5  # pixels above/below a separator line that count as a grab

    def _separator_y_positions(self) -> list[tuple[float, str, int]]:
        """Return (y, kind, index) for each draggable panel separator.

        kind is one of: 'vol' (top of volume), 'panel' (top of a sub-panel).
        index is -1 for volume, or the sub-panel index.
        """
        main_rect, vol_rect, panel_rects = self._layout_panels()
        seps: list[tuple[float, str, int]] = []
        if vol_rect is not None and vol_rect.height() > 0:
            seps.append((vol_rect.y(), "vol", -1))
        for i, pr in enumerate(panel_rects):
            if pr.height() > 0:
                seps.append((pr.y(), "panel", i))
        return seps

    def _chart_rect(self) -> QRectF:
        """Return the rectangle for the main (OHLC) chart area."""
        return self._layout_panels()[0]

    def _volume_rect(self) -> QRectF:
        """Return the rectangle for the volume sub-chart."""
        vr = self._layout_panels()[1]
        if vr is None:
            # Fallback: position below main with zero height
            mr = self._layout_panels()[0]
            return QRectF(mr.x(), mr.y() + mr.height(), mr.width(), 0.0)
        return vr

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _visible_range(self) -> tuple[int, int]:
        s = max(0, self._view_start - 1)
        e = min(self._n - 1, self._view_end + 1)
        return s, e

    def _price_range(self, s: int, e: int) -> tuple[float, float]:
        lo = float(np.nanmin(self._lows[s : e + 1]))
        hi = float(np.nanmax(self._highs[s : e + 1]))
        # include visible addplot data in range
        for ap_idx, ap in enumerate(self._addplots):
            if not self._addplot_visible[ap_idx]:
                continue
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
        # include visible MAV data
        for mav_idx, ma in enumerate(self._mav_data):
            if not self._mav_visible[mav_idx]:
                continue
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
        count = self._view_end - self._view_start + 1
        if count <= 1:
            return rect.x() + rect.width() / 2
        bar_spacing = rect.width() / (count - 1)
        return rect.x() + (idx - self._view_start - self._pan_offset) * bar_spacing

    def _index_for_x(self, x: float, rect: QRectF, s: int, e: int) -> int:
        count = self._view_end - self._view_start + 1
        if count <= 1:
            return self._view_start
        bar_spacing = rect.width() / (count - 1)
        return int(round(self._view_start + self._pan_offset + (x - rect.x()) / bar_spacing))

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
    # Aggregation (level-of-detail)
    # ------------------------------------------------------------------

    def _compute_agg_factor(self) -> int:
        """How many raw bars to merge into one display bar."""
        count = self._view_end - self._view_start + 1
        if count <= self.CANDLE_BUDGET:
            return 1
        return math.ceil(count / self.CANDLE_BUDGET)

    @staticmethod
    def _agg_1d(data: np.ndarray, factor: int, n: int,
                method: str = "last") -> np.ndarray:
        """Downsample a 1-D array by groups of *factor* elements."""
        if len(data) < n:
            tmp = np.full(n, np.nan)
            tmp[: len(data)] = data
            data = tmp
        else:
            data = data[:n]
        n_agg = math.ceil(n / factor)
        pad = n_agg * factor - n
        if method == "sum":
            arr = np.concatenate([np.where(np.isnan(data), 0.0, data),
                                  np.zeros(pad)]) if pad else \
                  np.where(np.isnan(data), 0.0, data)
            return arr.reshape(n_agg, factor).sum(axis=1)
        if method == "mean":
            padded = np.concatenate([data, np.full(pad, np.nan)]) if pad else data
            reshaped = padded.reshape(n_agg, factor)
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(reshaped, axis=1)
        # "last" — take the value at the last raw bar of each group
        ends = np.minimum(np.arange(n_agg) * factor + factor, n) - 1
        return data[ends]

    @staticmethod
    def _agg_1d_scatter(data: np.ndarray, factor: int,
                        n: int) -> np.ndarray:
        """Downsample scatter: keep first non-NaN per group."""
        if len(data) < n:
            tmp = np.full(n, np.nan)
            tmp[: len(data)] = data
            data = tmp
        else:
            data = data[:n]
        n_agg = math.ceil(n / factor)
        result = np.full(n_agg, np.nan)
        starts = np.arange(n_agg) * factor
        ends = np.minimum(starts + factor, n)
        for i in range(n_agg):
            chunk = data[starts[i] : ends[i]]
            mask = ~np.isnan(chunk)
            if mask.any():
                result[i] = chunk[mask][0]
        return result

    def _enter_aggregated_mode(self, factor: int) -> dict[str, Any]:
        """Replace internal arrays with aggregated versions for painting."""
        n = self._n
        n_agg = math.ceil(n / factor)
        pad = n_agg * factor - n

        saved: dict[str, Any] = {
            "opens": self._opens, "highs": self._highs,
            "lows": self._lows, "closes": self._closes,
            "volumes": self._volumes, "dates": self._dates,
            "n": n, "view_start": self._view_start,
            "view_end": self._view_end, "pan_offset": self._pan_offset,
            "mav_data": self._mav_data,
            "addplots": self._addplots,
            "sub_panels": self._sub_panels,
        }

        starts = np.arange(n_agg) * factor
        ends = np.minimum(starts + factor, n)

        # ---- OHLCV ------------------------------------------------------------
        self._opens = saved["opens"][starts]
        self._closes = saved["closes"][ends - 1]

        if pad:
            h = np.concatenate([saved["highs"], np.full(pad, -np.inf)])
            lo = np.concatenate([saved["lows"], np.full(pad, np.inf)])
        else:
            h, lo = saved["highs"].copy(), saved["lows"].copy()
        self._highs = h.reshape(n_agg, factor).max(axis=1)
        self._lows = lo.reshape(n_agg, factor).min(axis=1)

        if saved["volumes"] is not None:
            v = np.concatenate([saved["volumes"], np.zeros(pad)]) if pad else saved["volumes"]
            self._volumes = v.reshape(n_agg, factor).sum(axis=1)

        if saved["dates"] is not None:
            self._dates = saved["dates"][starts]

        # ---- view state -------------------------------------------------------
        self._n = n_agg
        self._view_start = saved["view_start"] // factor
        self._view_end = min(saved["view_end"] // factor, n_agg - 1)
        self._pan_offset = saved["pan_offset"] / factor

        # ---- MAVs -------------------------------------------------------------
        self._mav_data = [
            self._agg_1d(ma, factor, n, "mean") for ma in saved["mav_data"]
        ]

        # ---- addplots ---------------------------------------------------------
        agg_addplots: list[dict[str, Any]] = []
        for ap in saved["addplots"]:
            agg = dict(ap)
            if ap["type"] == "segments":
                agg_segs: list[tuple[int, np.ndarray]] = []
                for seg_start, seg_vals in ap["data"]:
                    a_s = seg_start // factor
                    seg_end_raw = seg_start + len(seg_vals) - 1
                    a_e = seg_end_raw // factor
                    a_len = a_e - a_s + 1
                    if a_len < 1:
                        continue
                    a_vals = np.empty(a_len)
                    for j in range(a_len):
                        raw_hi = min((a_s + j + 1) * factor - 1,
                                     seg_end_raw) - seg_start
                        a_vals[j] = seg_vals[raw_hi]
                    agg_segs.append((a_s, a_vals))
                agg["data"] = agg_segs
            elif ap["type"] == "scatter":
                agg["data"] = self._agg_1d_scatter(ap["data"], factor, n)
            else:
                agg["data"] = self._agg_1d(ap["data"], factor, n, "mean")
            agg_addplots.append(agg)
        self._addplots = agg_addplots

        # ---- sub-panels -------------------------------------------------------
        agg_panels: list[SubPanel] = []
        for panel in saved["sub_panels"]:
            if panel.skip_aggregation:
                agg_panels.append(panel)
                continue
            method = "sum" if panel.panel_type == "histogram" else "mean"
            agg_data = self._agg_1d(panel.data, factor, n, method)
            agg_paps: list[dict[str, Any]] = []
            for pap in panel.addplots:
                agg_pap = dict(pap)
                if pap["type"] == "scatter":
                    agg_pap["data"] = self._agg_1d_scatter(
                        pap["data"], factor, n)
                elif pap["type"] == "segments":
                    agg_segs: list[tuple[int, np.ndarray]] = []
                    for seg_start, seg_vals in pap["data"]:
                        a_s = seg_start // factor
                        seg_end_raw = seg_start + len(seg_vals) - 1
                        a_e = seg_end_raw // factor
                        a_len = a_e - a_s + 1
                        if a_len < 1:
                            continue
                        a_vals = np.empty(a_len)
                        for j in range(a_len):
                            raw_hi = min((a_s + j + 1) * factor - 1,
                                         seg_end_raw) - seg_start
                            a_vals[j] = seg_vals[raw_hi]
                        agg_segs.append((a_s, a_vals))
                    agg_pap["data"] = agg_segs
                else:
                    agg_pap["data"] = self._agg_1d(pap["data"], factor, n, "mean")
                agg_paps.append(agg_pap)
            agg_panels.append(SubPanel(
                data=agg_data,
                ylabel=panel.ylabel,
                height_ratio=panel.height_ratio,
                color=panel.color,
                panel_type=panel.panel_type,
                width=panel.width,
                linestyle=panel.linestyle,
                alpha=panel.alpha,
                visible=panel.visible,
                addplots=agg_paps,
                zero_centered=panel.zero_centered,
            ))
        self._sub_panels = agg_panels

        # store original view info for skip_aggregation panels
        self._agg_orig_view = (saved["view_start"], saved["view_end"],
                               saved["pan_offset"])

        return saved

    def _exit_aggregated_mode(self, saved: dict[str, Any]) -> None:
        """Restore original data after aggregated painting."""
        self._opens = saved["opens"]
        self._highs = saved["highs"]
        self._lows = saved["lows"]
        self._closes = saved["closes"]
        self._volumes = saved["volumes"]
        self._dates = saved["dates"]
        self._n = saved["n"]
        self._view_start = saved["view_start"]
        self._view_end = saved["view_end"]
        self._pan_offset = saved["pan_offset"]
        self._mav_data = saved["mav_data"]
        self._addplots = saved["addplots"]
        self._sub_panels = saved["sub_panels"]
        self._agg_orig_view = None

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

        factor = self._compute_agg_factor()
        saved = self._enter_aggregated_mode(factor) if factor > 1 else None
        try:
            self._paint_impl(painter)
        finally:
            if saved is not None:
                self._exit_aggregated_mode(saved)

    def _paint_impl(self, painter: QPainter) -> None:
        s, e = self._visible_range()
        if s > e:
            return

        main_rect, vol_rect, panel_rects = self._layout_panels()
        plo, phi = self._price_range(s, e)
        count = self._view_end - self._view_start + 1
        cw = self._candle_width(main_rect, count)

        style = self._style
        alpha = style.get("alpha", 1.0)

        # --- grid ---------------------------------------------------------------
        self._draw_grid(painter, main_rect, plo, phi, style)

        # --- candles / bars / line ----------------------------------------------
        if self._chart_type in ("candle", "candlestick"):
            self._draw_candles(painter, main_rect, s, e, plo, phi, cw, style, alpha)
        elif self._chart_type in ("ohlc", "ohlc_bars", "bars"):
            self._draw_ohlc(painter, main_rect, s, e, plo, phi, cw, style, alpha)
        elif self._chart_type == "hollow":
            self._draw_hollow_candles(painter, main_rect, s, e, plo, phi, cw, style, alpha)
        elif self._chart_type == "line":
            self._draw_line(painter, main_rect, s, e, plo, phi, style)
        else:
            self._draw_candles(painter, main_rect, s, e, plo, phi, cw, style, alpha)

        # --- moving averages ----------------------------------------------------
        self._draw_mavs(painter, main_rect, s, e, plo, phi)

        # --- addplots -----------------------------------------------------------
        self._draw_addplots(painter, main_rect, s, e, plo, phi)

        # --- price axis (right) -------------------------------------------------
        self._draw_price_axis(painter, main_rect, plo, phi, style)

        # --- volume -------------------------------------------------------------
        if self._show_volume and vol_rect is not None:
            vlo, vhi = self._volume_range(s, e)
            self._draw_volume(painter, vol_rect, s, e, vlo, vhi, cw, style, alpha)
            self._draw_volume_axis(painter, vol_rect, vlo, vhi, style)

        # --- user sub-panels ----------------------------------------------------
        for panel, prect in zip(self._sub_panels, panel_rects):
            if panel.skip_aggregation and self._agg_orig_view is not None:
                # Temporarily restore original view coords so _x_for_index
                # maps segment indices correctly at full resolution.
                orig_s, orig_e, orig_off = self._agg_orig_view
                agg_s, agg_e, agg_off = (
                    self._view_start, self._view_end, self._pan_offset)
                self._view_start = orig_s
                self._view_end = orig_e
                self._pan_offset = orig_off
                self._draw_sub_panel(painter, panel, prect, orig_s, orig_e)
                self._view_start = agg_s
                self._view_end = agg_e
                self._pan_offset = agg_off
            else:
                self._draw_sub_panel(painter, panel, prect, s, e)

        # --- date axis ----------------------------------------------------------
        all_sub_rects = [
            r for r in ([vol_rect] + panel_rects) if r is not None and r.height() > 0
        ]
        bottom_rect = all_sub_rects[-1] if all_sub_rects else main_rect
        self._draw_date_axis(painter, bottom_rect, s, e, style)

        # --- title --------------------------------------------------------------
        if self._title:
            painter.setPen(QPen(_qcolor(style.get("text_color", "#333"))))
            font = QFont("Segoe UI", 11)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QRectF(self.MARGIN_LEFT, 2, main_rect.width(), self.MARGIN_TOP - 4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._title,
            )

        # --- crosshair ----------------------------------------------------------
        if self._mouse_pos is not None:
            all_rects = [main_rect] + ([vol_rect] if vol_rect else []) + [
                r for r in panel_rects if r.height() > 0
            ]
            self._draw_crosshair(painter, main_rect, all_rects, s, e, plo, phi, style)

        # --- legend -------------------------------------------------------------
        self._draw_legend(painter, main_rect, style)

    # ---- legend ----

    def _draw_eye_icon(self, p: QPainter, cx: float, cy: float,
                       size: float, open_: bool, color: QColor) -> None:
        """Draw an open or closed eye icon centred at (cx, cy)."""
        hw = size * 0.50   # half-width of the outer oval
        hh = size * 0.30   # half-height of the outer oval
        p.setPen(QPen(color, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if open_:
            p.drawEllipse(QPointF(cx, cy), hw, hh)
            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), hw * 0.36, hw * 0.36)
        else:
            # closed: a flat horizontal line with short downward lash ticks
            p.setPen(QPen(color, 1.2))
            p.drawLine(QPointF(cx - hw, cy), QPointF(cx + hw, cy))
            tick = hh * 0.8
            for tx in (cx - hw * 0.5, cx, cx + hw * 0.5):
                p.drawLine(QPointF(tx, cy), QPointF(tx, cy + tick))

    def _draw_legend(self, p: QPainter, rect: QRectF, style: dict) -> None:
        """Draw a vertical legend with per-series eye-icon visibility toggles."""
        # --- build entry list: (color, label, pen_style, kind, kind_idx, visible) ---
        entries: list[tuple[QColor, str, Qt.PenStyle, str, int, bool]] = []

        mav_colors = style.get("mavcolors", _MAV_COLORS)
        for idx, period in enumerate(self._mavs):
            entries.append((
                _qcolor(mav_colors[idx % len(mav_colors)]),
                f"MA {period}",
                Qt.PenStyle.SolidLine,
                "mav", idx,
                self._mav_visible[idx],
            ))

        for ap_idx, ap in enumerate(self._addplots):
            label = ap.get("ylabel")
            if not label:
                continue
            color = _qcolor(ap.get("color") or "#1f77b4", ap.get("alpha", 1.0))
            ls = ap.get("linestyle", "-")
            if ls in ("--", "dashed"):
                ps = Qt.PenStyle.DashLine
            elif ls in ("-.", "dashdot"):
                ps = Qt.PenStyle.DashDotLine
            elif ls in (":", "dotted"):
                ps = Qt.PenStyle.DotLine
            else:
                ps = Qt.PenStyle.SolidLine
            entries.append((color, label, ps, "addplot", ap_idx,
                            self._addplot_visible[ap_idx]))

        if self._show_volume:
            vol_color = _qcolor(
                style.get("volume_up", style.get("up_color", "#26a69a")), 0.7
            )
            entries.append((vol_color, "Volume", Qt.PenStyle.SolidLine,
                            "volume", 0, self._volume_visible))

        # Collect sub-panel entries
        for pi, panel in enumerate(self._sub_panels):
            entries.append((
                _qcolor(panel.color, panel.alpha),
                panel.ylabel,
                Qt.PenStyle.SolidLine,
                "panel", pi,
                panel.visible,
            ))

        if not entries:
            self._legend_rect = None
            self._legend_hit_areas = []
            return

        # --- layout constants ---------------------------------------------------
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        fm = QFontMetrics(font)

        eye_sz   = 10.0
        swatch_w = 18.0
        gap      = 5.0
        pad_x    = 8.0
        pad_y    = 5.0
        row_h    = max(float(fm.height()), eye_sz) + 4.0
        row_gap  = 2.0

        max_lw  = max(fm.horizontalAdvance(e[1]) for e in entries)
        total_w = pad_x * 2 + eye_sz + gap + swatch_w + gap + max_lw
        total_h = pad_y * 2 + len(entries) * row_h + max(0, len(entries) - 1) * row_gap

        box_x = rect.x() + 4.0
        box_y = rect.y() + 4.0

        # --- background box -----------------------------------------------------
        p.setPen(QPen(_qcolor(style.get("grid_color", "#e0e0e0"), 0.60), 1))
        p.setBrush(QBrush(_qcolor(style.get("bg_color", "#ffffff"), 0.88)))
        legend_rect = QRectF(box_x, box_y, total_w, total_h)
        p.drawRoundedRect(legend_rect, 4, 4)
        self._legend_rect      = legend_rect
        self._legend_hit_areas = []

        text_color = _qcolor(style.get("text_color", "#333"))
        hover_fill = _qcolor(style.get("text_color", "#333"), 0.07)

        # --- draw rows ----------------------------------------------------------
        for row_idx, (color, label, pen_style, kind, kind_idx, visible) in enumerate(entries):
            row_y = box_y + pad_y + row_idx * (row_h + row_gap)
            cy    = row_y + row_h / 2.0

            row_rect = QRectF(box_x + 2, row_y, total_w - 4, row_h)
            eye_rect = QRectF(box_x + pad_x, cy - eye_sz / 2, eye_sz, eye_sz)

            # row hover highlight
            if row_idx == self._legend_hover_idx:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(hover_fill))
                p.drawRoundedRect(row_rect, 3, 3)

            cx = box_x + pad_x

            # eye icon
            eye_c = QColor(text_color)
            eye_c.setAlphaF(1.0 if visible else 0.30)
            self._draw_eye_icon(p, cx + eye_sz / 2, cy, eye_sz, visible, eye_c)
            cx += eye_sz + gap

            # colour swatch (dimmed when hidden)
            sc = QColor(color)
            sc.setAlphaF(sc.alphaF() * (1.0 if visible else 0.25))
            pen = QPen(sc, 2.0)
            pen.setStyle(pen_style)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(QPointF(cx, cy), QPointF(cx + swatch_w, cy))
            cx += swatch_w + gap

            # label (dimmed when hidden)
            lc = QColor(text_color)
            lc.setAlphaF(1.0 if visible else 0.35)
            p.setPen(QPen(lc))
            p.drawText(QPointF(cx, cy + fm.ascent() / 2.0 - 1), label)

            self._legend_hit_areas.append((row_rect, eye_rect, kind, kind_idx))

    # ---- grid ----
    def _draw_grid(self, p: QPainter, rect: QRectF, lo: float, hi: float, style: dict) -> None:
        pen = QPen(_qcolor(style.get("grid_color", "#e0e0e0")), 1, Qt.PenStyle.DotLine)
        p.setPen(pen)
        n_lines = max(1, min(6, int(rect.height() / 30)))
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
            if not self._mav_visible[idx]:
                continue
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
        for ap_idx, ap in enumerate(self._addplots):
            if not self._addplot_visible[ap_idx]:
                continue
            color_str = ap.get("color") or "#1f77b4"
            ap_alpha = ap.get("alpha", 1.0)
            color = _qcolor(color_str, ap_alpha)

            if ap["type"] == "segments":
                self._draw_segments(p, rect, s, e, lo, hi, ap, color)
            elif ap["type"] == "scatter":
                self._draw_scatter(p, rect, s, e, lo, hi, ap, color)
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

    def _draw_scatter(self, p: QPainter, rect: QRectF, s: int, e: int,
                      lo: float, hi: float, ap: dict, color: QColor) -> None:
        """Draw scatter markers with support for 'o', '^', and 'v' shapes."""
        data = ap["data"]
        marker_size = ap.get("markersize", 50)
        radius = math.sqrt(marker_size) / 2
        marker = ap.get("marker", "o")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        for i in range(s, e + 1):
            val = data[i] if i < len(data) else float("nan")
            if np.isnan(val):
                continue
            x = self._x_for_index(i, rect, s, e)
            y = self._y_for_price(val, rect, lo, hi)
            if marker == "^":
                tri = QPolygonF([
                    QPointF(x, y - radius),
                    QPointF(x - radius, y + radius),
                    QPointF(x + radius, y + radius),
                ])
                p.drawPolygon(tri)
            elif marker == "v":
                tri = QPolygonF([
                    QPointF(x, y + radius),
                    QPointF(x - radius, y - radius),
                    QPointF(x + radius, y - radius),
                ])
                p.drawPolygon(tri)
            else:
                p.drawEllipse(QPointF(x, y), radius, radius)

    # ---- volume ----
    def _draw_volume(self, p: QPainter, rect: QRectF, s: int, e: int,
                     vlo: float, vhi: float, cw: float, style: dict, alpha: float) -> None:
        if self._volumes is None or not self._volume_visible:
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
        label_h = fm.height()
        n_ticks = max(1, min(6, int(rect.height() / max(label_h * 1.8, 1))))
        x_label = rect.x() + rect.width() + 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            price = hi - frac * (hi - lo)
            y = rect.y() + frac * rect.height()
            label = f"{price:,.2f}"
            p.drawText(QPointF(x_label, y + fm.ascent() / 2), label)

    def _draw_volume_axis(self, p: QPainter, rect: QRectF, vlo: float, vhi: float, style: dict) -> None:
        if not self._volume_visible:
            return
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

    # ---- sub-panels ----

    def _panel_y_range(self, panel: SubPanel, s: int, e: int) -> tuple[float, float] | None:
        """Compute the Y range for a sub-panel, or None if all data is NaN."""
        seg = panel.data[s : e + 1]
        valid = seg[~np.isnan(seg)]
        has_data = len(valid) > 0
        lo = float(np.nanmin(valid)) if has_data else float("inf")
        hi = float(np.nanmax(valid)) if has_data else float("-inf")
        if panel.panel_type == "histogram":
            lo = min(lo, 0.0)
        # also include any addplot data on this panel
        for ap in panel.addplots:
            if ap["type"] == "segments":
                for seg_start, seg_data in ap["data"]:
                    seg_end = seg_start + len(seg_data)
                    lo_idx = max(s, seg_start)
                    hi_idx = min(e, seg_end - 1)
                    if lo_idx > hi_idx:
                        continue
                    chunk = seg_data[lo_idx - seg_start : hi_idx - seg_start + 1]
                    seg_valid = chunk[~np.isnan(chunk)]
                    if len(seg_valid):
                        lo = min(lo, float(np.nanmin(seg_valid)))
                        hi = max(hi, float(np.nanmax(seg_valid)))
                        has_data = True
            else:
                ap_seg = ap["data"][s : e + 1]
                ap_valid = ap_seg[~np.isnan(ap_seg)]
                if len(ap_valid):
                    lo = min(lo, float(np.nanmin(ap_valid)))
                    hi = max(hi, float(np.nanmax(ap_valid)))
                    has_data = True
        if not has_data:
            return None
        pad = (hi - lo) * 0.05 if hi != lo else 1.0
        if panel.zero_centered:
            extreme = max(abs(lo), abs(hi))
            return -(extreme + pad), extreme + pad
        return lo - pad, hi + pad

    def _draw_sub_panel(self, p: QPainter, panel: SubPanel,
                        rect: QRectF, s: int, e: int) -> None:
        """Draw a single sub-panel below the main chart."""
        if not panel.visible or rect.height() < 1:
            return
        yr = self._panel_y_range(panel, s, e)
        if yr is None:
            return
        lo, hi = yr

        style = self._style

        # separator + grid
        p.setPen(QPen(_qcolor(style.get("grid_color", "#e0e0e0")), 1))
        p.drawLine(QPointF(rect.x(), rect.y()), QPointF(rect.x() + rect.width(), rect.y()))
        self._draw_grid(p, rect, lo, hi, style)

        color = _qcolor(panel.color, panel.alpha)
        if panel.panel_type == "histogram":
            self._draw_panel_histogram(p, panel, rect, s, e, lo, hi, color)
        else:
            self._draw_panel_line(p, panel, rect, s, e, lo, hi, color)

        # horizontal zero line
        if panel.zero_centered and lo < 0 < hi:
            zero_y = self._y_for_price(0.0, rect, lo, hi)
            p.setPen(QPen(_qcolor(style.get("grid_color", "#e0e0e0")), 1,
                          Qt.PenStyle.DashLine))
            p.drawLine(QPointF(rect.x(), zero_y),
                       QPointF(rect.x() + rect.width(), zero_y))

        # overlays on this panel
        self._draw_addplot_list(p, panel.addplots, rect, s, e, lo, hi)

        # price axis
        self._draw_price_axis(p, rect, lo, hi, style)

        # ylabel label
        p.setPen(QPen(_qcolor(style.get("text_color", "#333"), 0.70)))
        font = QFont("Segoe UI", 7)
        p.setFont(font)
        fm = QFontMetrics(font)
        p.drawText(QPointF(rect.x() + 4, rect.y() + fm.ascent() + 3), panel.ylabel)

    def _draw_panel_line(self, p: QPainter, panel: SubPanel, rect: QRectF,
                         s: int, e: int, lo: float, hi: float,
                         color: QColor) -> None:
        lw = panel.width
        pen = QPen(color, lw)
        ls = panel.linestyle
        if ls in ("--", "dashed"):
            pen.setStyle(Qt.PenStyle.DashLine)
        elif ls in ("-.", "dashdot"):
            pen.setStyle(Qt.PenStyle.DashDotLine)
        elif ls in (":", "dotted"):
            pen.setStyle(Qt.PenStyle.DotLine)
        p.setPen(pen)
        prev: QPointF | None = None
        for i in range(s, e + 1):
            val = panel.data[i] if i < len(panel.data) else float("nan")
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

    def _draw_panel_histogram(self, p: QPainter, panel: SubPanel, rect: QRectF,
                              s: int, e: int, lo: float, hi: float,
                              color: QColor) -> None:
        count = self._view_end - self._view_start + 1
        cw = self._candle_width(rect, count)
        zero_y = self._y_for_price(0.0, rect, lo, hi)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        for i in range(s, e + 1):
            val = panel.data[i] if i < len(panel.data) else float("nan")
            if np.isnan(val):
                continue
            x   = self._x_for_index(i, rect, s, e)
            y   = self._y_for_price(val, rect, lo, hi)
            top = min(y, zero_y)
            bh  = max(abs(y - zero_y), 1.0)
            p.drawRect(QRectF(x - cw / 2, top, cw, bh))

    def _draw_addplot_list(
        self,
        p: QPainter,
        addplots: list[dict[str, Any]],
        rect: QRectF,
        s: int,
        e: int,
        lo: float,
        hi: float,
    ) -> None:
        """Draw a list of addplot dicts against an arbitrary rect + Y range."""
        for ap in addplots:
            color_str = ap.get("color") or "#1f77b4"
            ap_alpha = ap.get("alpha", 1.0)
            color = _qcolor(color_str, ap_alpha)

            if ap["type"] == "segments":
                self._draw_segments(p, rect, s, e, lo, hi, ap, color)
            elif ap["type"] == "scatter":
                self._draw_scatter(p, rect, s, e, lo, hi, ap, color)
            else:
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
                prev: QPointF | None = None
                data = ap["data"]
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

    # ---- crosshair ----
    def _draw_crosshair(
        self,
        p: QPainter,
        rect: QRectF,
        all_rects: list[QRectF],
        s: int,
        e: int,
        lo: float,
        hi: float,
        style: dict,
    ) -> None:
        mx = self._mouse_pos.x()  # type: ignore[union-attr]
        my = self._mouse_pos.y()  # type: ignore[union-attr]

        full_top    = all_rects[0].y()
        full_bottom = all_rects[-1].y() + all_rects[-1].height()

        if not (rect.x() <= mx <= rect.x() + rect.width() and full_top <= my <= full_bottom):
            return

        pen = QPen(_qcolor(style.get("text_color", "#333"), 0.4), 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(rect.x(), my), QPointF(rect.x() + rect.width(), my))
        p.drawLine(QPointF(mx, full_top), QPointF(mx, full_bottom))

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
        self._pan_offset = 0.0
        delta = event.angleDelta().y()
        rect = self._chart_rect()
        count = self._view_end - self._view_start + 1

        # Fraction of visible range under the cursor (0 = left edge, 1 = right)
        mx = event.position().x()
        anchor = max(0.0, min(1.0, (mx - rect.x()) / rect.width())) if rect.width() > 0 else 0.5

        zoom_amount = max(1, int(count * 0.1))

        if delta > 0:
            # zoom in — remove bars proportionally around the cursor
            remove_left  = max(0, int(round(zoom_amount * anchor)))
            remove_right = max(0, zoom_amount - remove_left)
            new_s = min(self._view_start + remove_left, self._view_end - 2)
            new_e = max(self._view_end - remove_right, new_s + 2)
        else:
            # zoom out — add bars proportionally around the cursor
            add_left  = max(0, int(round(zoom_amount * anchor)))
            add_right = max(0, zoom_amount - add_left)
            new_s = max(0, self._view_start - add_left)
            new_e = min(self._n - 1, self._view_end + add_right)

        self._view_start = new_s
        self._view_end   = new_e
        self.rangeChanged.emit(self._view_start, self._view_end)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            # --- legend eye-icon click: toggle visibility, suppress drag ---
            for row_rect, eye_rect, kind, kind_idx in self._legend_hit_areas:
                if eye_rect.contains(pos):
                    if kind == "mav":
                        self._mav_visible[kind_idx] = not self._mav_visible[kind_idx]
                    elif kind == "addplot":
                        self._addplot_visible[kind_idx] = not self._addplot_visible[kind_idx]
                    elif kind == "volume":
                        self._volume_visible = not self._volume_visible
                    elif kind == "panel":
                        self._sub_panels[kind_idx].visible = not self._sub_panels[kind_idx].visible
                    self.update()
                    return
            # --- separator resize start ---
            grab = self._SEPARATOR_GRAB
            for sep_idx, (sy, skind, sidx) in enumerate(self._separator_y_positions()):
                if abs(pos.y() - sy) <= grab:
                    self._resizing_sep = sep_idx
                    self._resize_last_y = pos.y()
                    return
            # --- normal drag start ---
            self._dragging = True
            self._drag_last_x = pos.x()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._resizing_sep = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        self._mouse_pos = pos

        # --- separator resize drag ---
        if self._resizing_sep is not None:
            dy = pos.y() - self._resize_last_y
            if dy == 0:
                self.update()
                return
            self._resize_last_y = pos.y()
            usable_h = self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM
            if usable_h <= 0:
                self.update()
                return
            ratio_delta = dy / usable_h
            seps = self._separator_y_positions()
            if self._resizing_sep >= len(seps):
                self._resizing_sep = None
                self.update()
                return
            _, skind, sidx = seps[self._resizing_sep]
            min_ratio = 0.03

            # Determine above/below regions and transfer height between them.
            # "below" is the panel whose top edge we're dragging.
            # "above" is the region immediately above that separator.
            def _get_ratio_below() -> float:
                if skind == "vol":
                    return self.VOLUME_RATIO
                return self._sub_panels[sidx].height_ratio

            def _set_ratio_below(v: float) -> None:
                if skind == "vol":
                    self.VOLUME_RATIO = v
                else:
                    self._sub_panels[sidx].height_ratio = v

            # Find what's above this separator
            sep_order: list[tuple[str, int]] = []
            if self._show_volume and self._volume_visible:
                sep_order.append(("vol", -1))
            for pi, pp in enumerate(self._sub_panels):
                if pp.visible:
                    sep_order.append(("panel", pi))
            # Find our position in the ordered list
            cur_pos = -1
            for ci, (ck, ci_idx) in enumerate(sep_order):
                if ck == skind and ci_idx == sidx:
                    cur_pos = ci
                    break
            if cur_pos <= 0:
                # Top-most sub-region: above is main chart (implicit).
                # Use effective (post-clamp) pixel sizes to compute the
                # resize and write back normalised ratios.  This avoids the
                # proportional-clamping artefact that otherwise prevents the
                # boundary from moving when total_sub > max_sub.
                main_now, vol_now, prects_now = self._layout_panels()
                eff_vol = (vol_now.height() / usable_h) if vol_now is not None else 0.0
                eff_panels = [pr.height() / usable_h for pr in prects_now]
                sub_total = eff_vol + sum(eff_panels)
                if sub_total <= 0:
                    self.update()
                    return
                new_sub_total = max(0.10, min(0.90, sub_total - ratio_delta))
                factor = new_sub_total / sub_total
                if self._show_volume and self._volume_visible:
                    self.VOLUME_RATIO = max(min_ratio, eff_vol * factor)
                for pi2, pp2 in enumerate(self._sub_panels):
                    if pp2.visible and pi2 < len(eff_panels):
                        pp2.height_ratio = max(min_ratio, eff_panels[pi2] * factor)
            else:
                # Transfer between adjacent panels.
                above_kind, above_idx = sep_order[cur_pos - 1]
                old_below = _get_ratio_below()
                if above_kind == "vol":
                    old_above = self.VOLUME_RATIO
                else:
                    old_above = self._sub_panels[above_idx].height_ratio
                new_below = max(min_ratio, old_below - ratio_delta)
                actual_delta = old_below - new_below
                new_above = max(min_ratio, old_above + actual_delta)
                actual_delta = new_above - old_above
                new_below = max(min_ratio, old_below - actual_delta)
                _set_ratio_below(new_below)
                if above_kind == "vol":
                    self.VOLUME_RATIO = new_above
                else:
                    self._sub_panels[above_idx].height_ratio = new_above
            self.update()
            return

        # --- legend hover ---
        if self._legend_rect is not None and self._legend_rect.contains(pos):
            self._legend_hover_idx = None
            for row_idx, (row_rect, eye_rect, kind, kind_idx) in enumerate(self._legend_hit_areas):
                if row_rect.contains(pos):
                    self._legend_hover_idx = row_idx
                    break
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            # Check if hovering near a separator
            grab = self._SEPARATOR_GRAB
            near_sep = False
            for sy, _, _ in self._separator_y_positions():
                if abs(pos.y() - sy) <= grab:
                    near_sep = True
                    break
            if near_sep:
                self.setCursor(Qt.CursorShape.SplitVCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self._legend_hover_idx = None

        if self._dragging:
            dx = pos.x() - self._drag_last_x
            self._drag_last_x = pos.x()
            rect = self._chart_rect()
            count = self._view_end - self._view_start + 1
            bar_spacing = rect.width() / max(count - 1, 1)
            self._pan_offset += -dx / bar_spacing

            # Shift view by whole bars when offset accumulates
            int_shift = int(self._pan_offset)
            if int_shift != 0:
                span = self._view_end - self._view_start
                new_s = max(0, min(self._view_start + int_shift, self._n - 1 - span))
                actual_shift = new_s - self._view_start
                self._view_start = new_s
                self._view_end = new_s + span
                self._pan_offset -= actual_shift
                self.rangeChanged.emit(self._view_start, self._view_end)

            # Clamp offset at data boundaries
            if self._view_start == 0:
                self._pan_offset = max(0.0, self._pan_offset)
            if self._view_end >= self._n - 1:
                self._pan_offset = min(0.0, self._pan_offset)

        self.update()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._mouse_pos = None
        self._legend_hover_idx = None
        self._resizing_sep = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        self.update()
