"""``make_addplot`` — mplfinance-compatible helper for additional data overlays."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


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
        ``'line'`` or ``'scatter'``.
    color : str or None
        Colour string (hex / named).
    width : float
        Line width (for ``type='line'``).
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
        (reserved for future use)

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

    valid_types = ("line", "scatter")
    if type not in valid_types:
        raise ValueError(f"addplot type must be one of {valid_types}, got '{type}'")

    # Normalise data to a 1-D numpy array
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError(
                "make_addplot received a DataFrame with more than one column. "
                "Pass a single Series or one-column DataFrame."
            )
        values = data.iloc[:, 0].values.astype(float)
    elif isinstance(data, pd.Series):
        values = data.values.astype(float)
    else:
        values = np.asarray(data, dtype=float)

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
