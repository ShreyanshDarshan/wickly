"""``make_addplot`` / ``make_panel`` — helpers for chart overlays and sub-panels."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SubPanel — data model for a synced sub-chart below the main candle chart
# ---------------------------------------------------------------------------

@dataclass
class SubPanel:
    """Configuration for a synced sub-chart panel below the main chart.

    Parameters
    ----------
    data : np.ndarray
        1-D array of y-values, same length as the OHLCV data.
    ylabel : str
        Label displayed at the top-left of the panel and in the legend.
    height_ratio : float
        Fraction of the widget's usable height allocated to this panel.
        Default is ``0.20`` (20 %).  The sum of all sub-panel ratios is
        clamped to ≤ 0.75 so the main chart always keeps at least 25 %.
    color : str
        Line / bar colour (hex or named).
    panel_type : str
        ``'line'`` (default) or ``'histogram'`` (bars from zero baseline,
        suitable for RSI with a zero-line, volume, or momentum).
    width : float
        Line or bar outline width.
    linestyle : str
        ``'-'``, ``'--'``, ``'-.'``, or ``':'`` (line panels only).
    alpha : float
        Opacity 0–1.
    visible : bool
        Initial visibility.  Can be toggled via the eye icon in the legend.
    addplots : list[dict]
        Optional extra line/scatter overlays on this panel, built with
        :func:`make_addplot`.  They are drawn in the panel's own Y range.
    """

    data: np.ndarray
    ylabel: str = "Value"
    height_ratio: float = 0.20
    color: str = "#1f77b4"
    panel_type: str = "line"   # 'line' | 'histogram'
    width: float = 1.5
    linestyle: str = "-"
    alpha: float = 1.0
    visible: bool = True
    addplots: list[dict[str, Any]] = field(default_factory=list)


def _normalise_1d(data: pd.Series | pd.DataFrame | list | np.ndarray) -> np.ndarray:
    """Convert various array-like inputs to a 1-D float64 numpy array."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                "make_addplot received a DataFrame with more than one column. "
                "Pass a single Series or one-column DataFrame."
            )
        return data.iloc[:, 0].values.astype(float)
    if isinstance(data, pd.Series):
        return data.values.astype(float)
    return np.asarray(data, dtype=float)


def make_addplot(
    data: pd.Series | pd.DataFrame | list | np.ndarray,
    *,
    type: str = "line",
    color: str | None = None,
    width: float = 1.5,
    alpha: float = 1.0,
    scatter: bool = False,
    marker: str = "o",
    markersize: float = 50,
    linestyle: str = "-",
    secondary_y: bool = False,
    panel: int | str = 0,
    ylabel: str | None = None,
    **_extra: Any,
) -> dict[str, Any]:
    """Build an *addplot* dict describing extra data to overlay on the chart.

    Mirrors the ``mplfinance.make_addplot()`` signature so users can switch
    libraries with minimal changes.

    Parameters
    ----------
    data : array-like
        The y-values to plot.  Must be the same length as the OHLCV DataFrame
        passed to ``plot()``.
    type : str
        ``'line'``, ``'scatter'``, or ``'segments'``.  When ``'segments'`` is
        used, *data* must be a list of ``(start_index, values)`` tuples — see
        :func:`make_segments` for a convenient builder.
    color : str or None
        Colour string (hex / named).
    width : float
        Line width (for ``type='line'`` or ``'segments'``).
    alpha : float
        Opacity 0-1.
    scatter : bool
        *Deprecated*. Use ``type='scatter'`` instead.
    marker : str
        Marker style when ``type='scatter'``.
    markersize : float
        Marker size when ``type='scatter'``.
    linestyle : str
        Matplotlib-style linestyle string.
    secondary_y : bool
        (reserved for future use)
    panel : int or str
        (reserved for future use — currently everything goes on main panel)
    ylabel : str or None
        Legend label shown in the chart's overlay legend.  If given, the
        overlay appears in the legend with a colour swatch; omit or pass
        ``None`` to hide the overlay from the legend.

    Returns
    -------
    dict
        A configuration dictionary consumed by ``wickly.plot(addplot=...)``.
    """
    if scatter:
        warnings.warn(
            "scatter=True is deprecated; use type='scatter' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        type = "scatter"  # noqa: A001  (shadows built-in on purpose)

    valid_types = ("line", "scatter", "segments")
    if type not in valid_types:
        raise ValueError(f"addplot type must be one of {valid_types}, got '{type}'")

    if type == "segments":
        # data must be a list of (start_index, array-like) tuples
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
            segments.append((int(start), _normalise_1d(vals)))
        return {
            "data": segments,
            "type": "segments",
            "color": color,
            "width": width,
            "alpha": alpha,
            "marker": marker,
            "markersize": markersize,
            "linestyle": linestyle,
            "secondary_y": secondary_y,
            "panel": panel,
            "ylabel": ylabel,
        }

    # --- line / scatter (original path) ---
    values = _normalise_1d(data)

    return {
        "data": values,
        "type": type,
        "color": color,
        "width": width,
        "alpha": alpha,
        "marker": marker,
        "markersize": markersize,
        "linestyle": linestyle,
        "secondary_y": secondary_y,
        "panel": panel,
        "ylabel": ylabel,
    }


def make_segments(
    segments: list[tuple[int, pd.Series | pd.DataFrame | list | np.ndarray]],
    *,
    color: str | None = None,
    width: float = 1.5,
    alpha: float = 1.0,
    linestyle: str = "-",
    ylabel: str | None = None,
) -> dict[str, Any]:
    """Build an addplot dict for multiple independent (possibly overlapping) line segments.

    This is a convenience wrapper around ``make_addplot(type='segments', ...)``.
    Use it when you need overlapping lines that cannot be represented by a
    single NaN-separated array — for example, Knoxville Divergence signals
    where a new divergence can start before a previous one ends.

    Parameters
    ----------
    segments : list of (start_index, values) tuples
        Each tuple is ``(start_index, y_values)`` where *start_index* is the
        bar index at which the segment begins and *y_values* is an array-like
        of consecutive y-values.  Segments may overlap freely.
    color : str or None
        Colour string (hex / named).
    width : float
        Line width.
    alpha : float
        Opacity 0-1.
    linestyle : str
        ``'-'``, ``'--'``, ``'-.'``, or ``':'``.
    ylabel : str or None
        Legend label shown in the chart's overlay legend.  If given, the
        overlay appears in the legend with a colour swatch.

    Returns
    -------
    dict
        A configuration dictionary consumed by ``wickly.plot(addplot=...)``.

    Examples
    --------
    >>> seg1 = (10, [50.1, 50.5, 51.0, 50.8])   # bars 10–13
    >>> seg2 = (12, [51.2, 51.5, 51.3])          # bars 12–14, overlaps seg1
    >>> apd = wickly.make_segments([seg1, seg2], color='#e91e63', ylabel='Divergence')
    >>> wickly.plot(df, addplot=[apd])
    """
    return make_addplot(
        segments,  # type: ignore[arg-type]
        type="segments",
        color=color,
        width=width,
        alpha=alpha,
        linestyle=linestyle,
        ylabel=ylabel,
    )


# ---------------------------------------------------------------------------
# make_panel — build a sub-chart panel beneath the main candle chart
# ---------------------------------------------------------------------------

def make_panel(
    data: pd.Series | pd.DataFrame | list | np.ndarray,
    *,
    ylabel: str = "Value",
    height_ratio: float = 0.20,
    color: str = "#1f77b4",
    panel_type: str = "line",
    width: float = 1.5,
    linestyle: str = "-",
    alpha: float = 1.0,
    addplot: list[dict[str, Any]] | dict[str, Any] | None = None,
) -> SubPanel:
    """Build a :class:`SubPanel` for display below the main candlestick chart.

    Sub-panels share the same zoomed and panned X axis as the main chart
    but have their own independent Y axis and Y range.

    Parameters
    ----------
    data : array-like
        1-D y-values of the same length as the OHLCV DataFrame passed to
        ``plot()``.
    ylabel : str
        Panel title shown in the top-left corner and in the master legend.
    height_ratio : float
        Fraction of total widget height given to this panel (default 0.20).
    color : str
        Line or bar colour.
    panel_type : str
        ``'line'`` for a continuous line, ``'histogram'`` for bars anchored
        at zero (useful for RSI, momentum, volume-like indicators).
    width : float
        Line / outline width in pixels.
    linestyle : str
        ``'-'``, ``'--'``, ``'-.'``, or ``':'`` (line panels only).
    alpha : float
        Opacity 0–1.
    addplot : dict or list[dict] or None
        Optional overlay(s) drawn on this panel (e.g. a horizontal RSI
        threshold line).  Build them with :func:`make_addplot`.

    Returns
    -------
    SubPanel
        Pass this to ``wickly.plot(panels=[panel])`` or
        ``widget.add_panel(panel)``.

    Examples
    --------
    >>> rsi = compute_rsi(df["Close"], period=14)
    >>> panel = wickly.make_panel(rsi, ylabel="RSI", color="#ff9800",
    ...                           panel_type="line", height_ratio=0.20)
    >>> wickly.plot(df, type="candle", volume=True, panels=[panel])
    """
    values = _normalise_1d(data)

    addplots_list: list[dict[str, Any]] = []
    if addplot is not None:
        if isinstance(addplot, dict):
            addplots_list = [addplot]
        else:
            addplots_list = list(addplot)

    valid_types = ("line", "histogram")
    if panel_type not in valid_types:
        raise ValueError(
            f"panel_type must be one of {valid_types}, got '{panel_type}'"
        )

    return SubPanel(
        data=values,
        ylabel=ylabel,
        height_ratio=height_ratio,
        color=color,
        panel_type=panel_type,
        width=width,
        linestyle=linestyle,
        alpha=alpha,
        addplots=addplots_list,
    )
