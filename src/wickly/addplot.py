"""``make_addplot`` — mplfinance-compatible helper for additional data overlays."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


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
