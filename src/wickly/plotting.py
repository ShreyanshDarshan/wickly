"""Top-level ``plot()`` function — mplfinance-compatible entry point."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from wickly._utils import check_and_prepare_data
from wickly.chart_widget import CandlestickWidget
from wickly.styles import _get_style


# ---------------------------------------------------------------------------
# Valid plot types (mirrors mplfinance)
# ---------------------------------------------------------------------------
_VALID_TYPES = {
    "candle", "candlestick",
    "ohlc", "ohlc_bars", "bars",
    "line",
    "hollow",
}


def _ensure_app() -> QApplication:
    """Return the running QApplication, creating one if necessary."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app  # type: ignore[return-value]


def plot(  # noqa: C901 — intentionally mirrors mplfinance's big kwargs set
    data: pd.DataFrame,
    **kwargs: Any,
) -> tuple[CandlestickWidget, dict[str, Any]] | None:
    """Plot an OHLCV DataFrame as an interactive candlestick chart.

    The signature intentionally mirrors ``mplfinance.plot()`` so that users
    can switch between libraries with minimal changes.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns Open, High, Low, Close (case-insensitive) and
        optionally Volume.  The index should be a ``DatetimeIndex``.

    Keyword arguments (all optional)
    --------------------------------
    type : str
        ``'candle'`` (default), ``'ohlc'``, ``'line'``, ``'hollow'``
    style : str or dict
        Style name (``'default'``, ``'charles'``, ``'mike'``, ``'yahoo'``,
        ``'classic'``, ``'nightclouds'``) or a dict from ``make_style``.
    volume : bool
        If ``True``, show a volume sub-chart.
    mav : int or tuple of ints
        Moving-average period(s).
    title : str
        Chart title.
    ylabel : str
        Y-axis label.
    figsize : tuple[int, int]
        ``(width, height)`` of the window in pixels.  Default ``(960, 600)``.
    columns : tuple[str, ...]
        Override column names — ``("Open","High","Low","Close","Volume")``.
    addplot : dict or list[dict]
        Additional plot(s) created with ``make_addplot()``.
    savefig : str
        If given, save the chart to this file path.
    returnfig : bool
        If ``True`` return ``(widget, axes_dict)`` instead of blocking.
    block : bool
        If ``True`` (default) block until the window is closed.

    Returns
    -------
    ``None`` when *block=True*; otherwise ``(CandlestickWidget, dict)``
    where ``dict`` maps axis names to internal references.
    """

    # --- resolve kwargs -------------------------------------------------------
    chart_type: str = kwargs.get("type", "ohlc")
    if chart_type not in _VALID_TYPES:
        raise ValueError(f"Invalid chart type '{chart_type}'. Valid: {sorted(_VALID_TYPES)}")

    style = _get_style(kwargs.get("style", None))
    show_volume: bool = kwargs.get("volume", False)
    mav = kwargs.get("mav", None)
    title: str | None = kwargs.get("title", None)
    ylabel: str = kwargs.get("ylabel", "Price")
    figsize: tuple[int, int] = kwargs.get("figsize", (960, 600))
    columns: tuple[str, ...] = kwargs.get("columns", ("Open", "High", "Low", "Close", "Volume"))
    savefig: str | None = kwargs.get("savefig", None)
    returnfig: bool = kwargs.get("returnfig", False)
    block: bool = kwargs.get("block", True)

    addplot_arg = kwargs.get("addplot", None)
    addplots: list[dict[str, Any]] = []
    if addplot_arg is not None:
        if isinstance(addplot_arg, dict):
            addplots = [addplot_arg]
        elif isinstance(addplot_arg, (list, tuple)):
            addplots = list(addplot_arg)

    # --- prepare data ---------------------------------------------------------
    opens, highs, lows, closes, volumes, dates = check_and_prepare_data(data, columns)

    if show_volume and volumes is None:
        raise ValueError("volume=True but no Volume column found in data.")

    # --- build widget ---------------------------------------------------------
    app = _ensure_app()

    widget = CandlestickWidget(
        dates=dates,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        chart_type=chart_type,
        style=style,
        show_volume=show_volume,
        mav=mav,
        title=title,
        ylabel=ylabel,
        addplots=addplots,
    )
    widget.resize(*figsize)
    widget.setWindowTitle(title or "Wickly Chart")

    # --- save to file ---------------------------------------------------------
    if savefig:
        # Must show briefly to let Qt compute layout, then grab
        widget.show()
        app.processEvents()
        widget.save(savefig)
        if not returnfig and block:
            widget.close()

    axes_dict: dict[str, Any] = {
        "main": widget,
    }

    if returnfig:
        # Prevent Qt from deleting the C++ object when the window is closed
        # so the caller can safely call widget.close() without a RuntimeError.
        widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        widget.show()
        return widget, axes_dict

    # --- show -----------------------------------------------------------------
    widget.show()

    if block:
        app.exec()
        return None

    return widget, axes_dict


# ---------------------------------------------------------------------------
# Live / animated chart
# ---------------------------------------------------------------------------

def live_plot(
    data: pd.DataFrame,
    **kwargs: Any,
) -> tuple[CandlestickWidget, dict[str, Any]]:
    """Create a non-blocking chart that can be updated with new data.

    Works exactly like ``plot(returnfig=True, block=False)`` but is more
    explicit about its intent: the returned widget is designed to be fed
    new bars via :meth:`~wickly.chart_widget.CandlestickWidget.append_data`
    or updated in-place via
    :meth:`~wickly.chart_widget.CandlestickWidget.update_last`.

    Parameters
    ----------
    data : pd.DataFrame
        Initial OHLCV DataFrame.
    **kwargs
        All keyword arguments accepted by :func:`plot` **except**
        ``returnfig`` and ``block`` (which are forced to ``True`` / ``False``).

    Returns
    -------
    (CandlestickWidget, dict)
        The live widget and an axes dict.

    Examples
    --------
    >>> widget, axes = wickly.live_plot(df, type='candle', volume=True)
    >>> # later, when a new bar arrives:
    >>> widget.append_data(new_dates, new_opens, new_highs,
    ...                    new_lows, new_closes)
    >>> # or update the last candle in real time:
    >>> widget.update_last(close=new_price,
    ...                    high=max(old_high, new_price))
    """
    kwargs["returnfig"] = True
    kwargs["block"] = False
    result = plot(data, **kwargs)
    # plot() always returns a tuple when returnfig=True
    assert result is not None
    return result
