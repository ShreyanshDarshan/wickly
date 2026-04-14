"""``wickly.bt.plot`` — top-level function for backtesting result charts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt

from wickly.plotting import _ensure_app
from wickly.bt.chart_widget import BacktestWidget


def plot(
    stats: pd.Series,
    *,
    data: pd.DataFrame | None = None,
    plot_equity: bool = True,
    plot_return: bool = False,
    plot_drawdown: bool = False,
    plot_volume: bool = True,
    plot_trades: bool = True,
    plot_pl: bool = True,
    smooth_equity: bool = False,
    relative_equity: bool = True,
    show_legend: bool = True,
    style: dict[str, Any] | str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (960, 600),
    addplot: dict[str, Any] | list[dict[str, Any]] | None = None,
    savefig: str | None = None,
    returnfig: bool = False,
    block: bool = True,
) -> tuple[BacktestWidget, dict[str, Any]] | None:
    """Plot backtest results as an interactive chart.

    Drop-in replacement for ``backtesting.Backtest.plot()`` using native
    PyQt6 rendering instead of Bokeh.

    Parameters
    ----------
    stats : pd.Series
        The result of ``Backtest.run()`` from the *backtesting* library.
    data : pd.DataFrame or None
        OHLCV data.  If *None*, extracted from the strategy stored in *stats*.
    plot_equity : bool
        Show an equity-curve sub-panel (default *True*).
    plot_return : bool
        Show a separate return-percentage sub-panel (default *False*).
    plot_drawdown : bool
        Show a drawdown sub-panel (default *False*).
    plot_volume : bool
        Show a volume sub-chart (default *True*).
    plot_trades : bool
        Draw entry→exit trade lines on the main chart (default *True*).
    plot_pl : bool
        Show a P/L histogram sub-panel (default *True*).
    smooth_equity : bool
        Only update the equity curve at trade-exit bars.
    relative_equity : bool
        Show equity as a percentage return from its initial value.
    show_legend : bool
        Display the chart legend.
    style : str or dict or None
        Visual style name or a dict from ``wickly.make_style()``.
    title : str or None
        Window title.
    figsize : tuple[int, int]
        ``(width, height)`` in pixels.  Default ``(960, 600)``.
    addplot : dict or list[dict] or None
        Additional overlays created with ``wickly.make_addplot()``.
    savefig : str or None
        If given, save the chart image to this file path.
    returnfig : bool
        If *True*, return ``(widget, axes_dict)`` instead of blocking.
    block : bool
        If *True* (default), block until the window is closed.

    Returns
    -------
    None when *block=True*; otherwise ``(BacktestWidget, dict)``.
    """
    # --- extract backtest data from stats ------------------------------------
    equity_curve = stats.get("_equity_curve", None)
    trades = stats.get("_trades", None)
    strategy = stats.get("_strategy", None)

    # Resolve OHLCV data
    if data is None:
        if strategy is not None:
            _sdata = getattr(strategy, "_data", None)
            if _sdata is not None:
                df = getattr(_sdata, "df", getattr(_sdata, "_df", None))
                if df is not None:
                    data = df
        if data is None:
            raise ValueError(
                "No OHLCV data provided and could not extract it from stats. "
                "Pass the data= argument explicitly."
            )

    # --- extract indicators from strategy ------------------------------------
    n_bars = len(data)
    indicators: list[dict[str, Any]] = []
    if strategy is not None:
        for ind in getattr(strategy, "_indicators", []):
            try:
                arr = np.asarray(ind, dtype=float)
                ind_name = str(getattr(ind, "name",
                                       getattr(ind, "_name", "Indicator")))
                ind_overlay = getattr(ind, "overlay",
                                      getattr(ind, "_overlay", None))
                ind_color = getattr(ind, "color",
                                    getattr(ind, "_color", None))
                ind_scatter = bool(getattr(ind, "scatter",
                                           getattr(ind, "_scatter", False)))

                # 2-D indicators (e.g. Bollinger Bands): shape is
                # (num_lines, n_bars).  Each row becomes a separate
                # indicator sharing the same attributes.
                if arr.ndim == 2:
                    n_lines = arr.shape[0]
                    for line_idx in range(n_lines):
                        line = arr[line_idx, :n_bars]
                        line_name = f"{ind_name}_{line_idx}" if n_lines > 1 else ind_name
                        indicators.append({
                            "data": line,
                            "name": line_name,
                            "overlay": ind_overlay,
                            "color": ind_color,
                            "scatter": ind_scatter,
                        })
                else:
                    indicators.append({
                        "data": arr[:n_bars],
                        "name": ind_name,
                        "overlay": ind_overlay,
                        "color": ind_color,
                        "scatter": ind_scatter,
                    })
            except (TypeError, ValueError):
                continue

    # --- normalise addplot arg ------------------------------------------------
    addplots: list[dict[str, Any]] = []
    if addplot is not None:
        if isinstance(addplot, dict):
            addplots = [addplot]
        else:
            addplots = list(addplot)

    # --- build widget ---------------------------------------------------------
    app = _ensure_app()

    widget = BacktestWidget(
        data=data,
        equity_curve=equity_curve,
        trades=trades,
        indicators=indicators or None,
        addplots=addplots or None,
        style=style,
        plot_equity=plot_equity,
        plot_return=plot_return,
        plot_drawdown=plot_drawdown,
        plot_volume=plot_volume,
        plot_trades=plot_trades,
        plot_pl=plot_pl,
        smooth_equity=smooth_equity,
        relative_equity=relative_equity,
        show_legend=show_legend,
        title=title,
    )
    widget.resize(*figsize)
    widget.setWindowTitle(title or "Backtest Results")

    # --- save to file ---------------------------------------------------------
    if savefig:
        widget.show()
        app.processEvents()
        widget.save(savefig)
        if not returnfig and block:
            widget.close()

    axes_dict: dict[str, Any] = {"main": widget}

    if returnfig:
        widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        widget.show()
        return widget, axes_dict

    # --- show -----------------------------------------------------------------
    widget.show()

    if block:
        app.exec()
        return None

    return widget, axes_dict
