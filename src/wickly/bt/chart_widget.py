"""BacktestWidget — CandlestickWidget subclass for backtesting results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wickly._utils import check_and_prepare_data
from wickly.addplot import SubPanel, make_addplot, make_segments, make_panel
from wickly.bt._indicator_registry import lookup as _lookup_indicator, OutputSpec
from wickly.chart_widget import CandlestickWidget
from wickly.styles import _get_style


# ---------------------------------------------------------------------------
# Colour palette for auto-colouring indicators
# ---------------------------------------------------------------------------

_INDICATOR_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_overlay(values: np.ndarray, closes: np.ndarray) -> bool:
    """Heuristic: indicator is an overlay if most values are near Close.

    Uses the same rule as backtesting.py — if >60 % of values are within
    40 % of the close price, it is classified as an overlay.
    """
    n = min(len(values), len(closes))
    values, closes = values[:n], closes[:n]
    valid = ~(np.isnan(values) | np.isnan(closes))
    if valid.sum() < 2:
        return False
    ratio = values[valid] / closes[valid]
    return bool(((ratio > 0.6) & (ratio < 1.4)).mean() > 0.6)


def _build_trade_segments(
    trades: pd.DataFrame,
    n: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(winner_addplot, loser_addplot)`` segment dicts."""
    if trades is None or trades.empty:
        return None, None

    win_segs: list[tuple[int, np.ndarray]] = []
    loss_segs: list[tuple[int, np.ndarray]] = []

    for _, t in trades.iterrows():
        entry_bar = int(t["EntryBar"])
        exit_bar = int(t["ExitBar"])
        entry_price = float(t["EntryPrice"])
        exit_price = float(t["ExitPrice"])
        pnl = float(t["PnL"])

        length = max(exit_bar - entry_bar + 1, 2)
        values = np.linspace(entry_price, exit_price, length)
        seg = (entry_bar, values)

        if pnl >= 0:
            win_segs.append(seg)
        else:
            loss_segs.append(seg)

    win_ap = (
        make_segments(win_segs, color="#26a69a", width=8.0, alpha=0.3,
                      linestyle=":", ylabel="Winning trades")
        if win_segs else None
    )
    loss_ap = (
        make_segments(loss_segs, color="#ef5350", width=8.0, alpha=0.3,
                      linestyle=":", ylabel="Losing trades")
        if loss_segs else None
    )
    return win_ap, loss_ap


def _build_pl_data(trades: pd.DataFrame, n: int) -> np.ndarray:
    """Build a sparse P/L array with values only at trade-exit bars."""
    pl = np.full(n, np.nan)
    if trades is None or trades.empty:
        return pl
    for _, t in trades.iterrows():
        exit_bar = int(t["ExitBar"])
        if 0 <= exit_bar < n:
            pl[exit_bar] = float(t["PnL"])
    return pl


def _build_pl_panel(trades: pd.DataFrame, n: int) -> SubPanel:
    """Build a P/L panel with sloped trade segments and directional markers.

    Each trade is a sloped line from ``(entry_bar, 0)`` to
    ``(exit_bar, PnL)``.  The exit endpoint has a triangle marker:
    up (``^``) for long trades, down (``v``) for short trades,
    coloured green for profitable and red for losing trades.
    """
    win_segs: list[tuple[int, np.ndarray]] = []
    loss_segs: list[tuple[int, np.ndarray]] = []
    # Scatter arrays for exit markers (4 groups)
    win_long = np.full(n, np.nan)
    win_short = np.full(n, np.nan)
    loss_long = np.full(n, np.nan)
    loss_short = np.full(n, np.nan)

    if trades is not None and not trades.empty:
        for _, t in trades.iterrows():
            entry_bar = int(t["EntryBar"])
            exit_bar = int(t["ExitBar"])
            pnl = float(t["PnL"])
            size = float(t.get("Size", 1))
            is_long = size > 0

            length = max(exit_bar - entry_bar + 1, 2)
            values = np.linspace(0.0, pnl, length)
            seg = (entry_bar, values)

            if pnl >= 0:
                win_segs.append(seg)
                if is_long:
                    if 0 <= exit_bar < n:
                        win_long[exit_bar] = pnl
                else:
                    if 0 <= exit_bar < n:
                        win_short[exit_bar] = pnl
            else:
                loss_segs.append(seg)
                if is_long:
                    if 0 <= exit_bar < n:
                        loss_long[exit_bar] = pnl
                else:
                    if 0 <= exit_bar < n:
                        loss_short[exit_bar] = pnl

    panel_addplots: list[dict] = []
    if win_segs:
        panel_addplots.append(make_segments(
            win_segs, color="#26a69a", width=1.0, linestyle="-",
        ))
    if loss_segs:
        panel_addplots.append(make_segments(
            loss_segs, color="#ef5350", width=1.0, linestyle="-",
        ))
    # Markers at exit: green triangle-up (long win), green triangle-down (short win),
    # red triangle-up (long loss), red triangle-down (short loss)
    if not np.all(np.isnan(win_long)):
        panel_addplots.append(make_addplot(
            win_long, type="scatter", color="#26a69a", marker="^", markersize=60,
        ))
    if not np.all(np.isnan(win_short)):
        panel_addplots.append(make_addplot(
            win_short, type="scatter", color="#26a69a", marker="v", markersize=60,
        ))
    if not np.all(np.isnan(loss_long)):
        panel_addplots.append(make_addplot(
            loss_long, type="scatter", color="#ef5350", marker="^", markersize=60,
        ))
    if not np.all(np.isnan(loss_short)):
        panel_addplots.append(make_addplot(
            loss_short, type="scatter", color="#ef5350", marker="v", markersize=60,
        ))

    return SubPanel(
        data=np.full(n, np.nan),
        ylabel="P/L",
        height_ratio=0.12,
        color="#7e57c2",
        panel_type="line",
        addplots=panel_addplots,
        aggregation_method="none",
        zero_centered=True,
    )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class BacktestWidget(CandlestickWidget):
    """Candlestick chart augmented with backtesting results.

    This is a thin wrapper around :class:`~wickly.chart_widget.CandlestickWidget`
    that converts equity curves, trades, and strategy indicators into the
    regular addplot / sub-panel primitives the base widget already renders.

    Parameters
    ----------
    parent : QWidget or None
        Optional parent widget.
    data : pd.DataFrame
        OHLCV DataFrame (same format as ``wickly.plot``).
    equity_curve : pd.DataFrame or None
        DataFrame with at least an ``Equity`` column (and optionally
        ``DrawdownPct``).  Typically ``stats['_equity_curve']``.
    trades : pd.DataFrame or None
        Trade log with ``EntryBar``, ``ExitBar``, ``EntryPrice``,
        ``ExitPrice``, ``PnL`` columns.  Typically ``stats['_trades']``.
    indicators : list of dict or None
        Each dict has keys ``data`` (array), ``name`` (str), and optionally
        ``overlay`` (bool), ``color`` (str), ``scatter`` (bool).
    addplots : list of dict or None
        User-supplied addplots to merge with auto-generated ones.
    style : str or dict or None
        Visual style.
    plot_equity / plot_return / plot_drawdown / plot_volume / plot_trades / plot_pl : bool
        Toggle individual chart panels.
    smooth_equity : bool
        Show equity changes only at trade-exit bars.
    relative_equity : bool
        Show equity as percentage return from the initial value.
    show_legend : bool
        Display the chart legend.
    title : str or None
        Window title.
    """

    def __init__(
        self,
        parent=None,
        *,
        data: pd.DataFrame,
        equity_curve: pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        indicators: list[dict[str, Any]] | None = None,
        addplots: list[dict[str, Any]] | None = None,
        style: dict[str, Any] | str | None = None,
        plot_equity: bool = True,
        plot_return: bool = False,
        plot_drawdown: bool = False,
        plot_volume: bool = True,
        plot_trades: bool = True,
        plot_pl: bool = True,
        smooth_equity: bool = False,
        relative_equity: bool = True,
        show_legend: bool = True,
        title: str | None = None,
    ):
        resolved_style = _get_style(style)
        resolved_style["alpha"] = resolved_style.get("alpha", 1.0)
        opens, highs, lows, closes, volumes, dates = check_and_prepare_data(data)
        n = len(closes)

        # --- build addplots and panels from backtest data ---------------------
        all_addplots: list[dict[str, Any]] = list(addplots) if addplots else []
        panels: list[SubPanel] = []

        # Trade entry→exit segments
        if plot_trades and trades is not None and not trades.empty:
            win_ap, loss_ap = _build_trade_segments(trades, n)
            if win_ap:
                all_addplots.append(win_ap)
            if loss_ap:
                all_addplots.append(loss_ap)

        # Equity curve panel
        if plot_equity and equity_curve is not None and "Equity" in equity_curve.columns:
            eq = equity_curve["Equity"].values.astype(float).copy()
            if smooth_equity:
                mask = np.diff(eq, prepend=eq[0]) != 0
                mask[0] = True
                eq_smooth = np.full_like(eq, np.nan)
                eq_smooth[mask] = eq[mask]
                last = eq[0]
                for i in range(len(eq_smooth)):
                    if np.isnan(eq_smooth[i]):
                        eq_smooth[i] = last
                    else:
                        last = eq_smooth[i]
                eq = eq_smooth
            # Build drawdown overlay if available
            eq_addplots: list[dict[str, Any]] = []
            if relative_equity:
                initial = eq[0] if eq[0] != 0 else 1.0
                eq = (eq / initial - 1) * 100 + 100

                if plot_drawdown:
                    # Running peak of eq; draw horizontal segments where
                    # the peak sits above the current equity curve.
                    peak = np.maximum.accumulate(eq[:n])
                    underwater = peak > eq[:n]
                    dd_segs: list[tuple[int, np.ndarray]] = []
                    i = 0
                    while i < n:
                        if underwater[i]:
                            start = i
                            level = peak[i]
                            while i < n and underwater[i] and peak[i] == level:
                                i += 1
                            dd_segs.append((start, np.full(i - start, level)))
                        else:
                            i += 1
                    if dd_segs:
                        eq_addplots.append(make_segments(
                            dd_segs, color="#e91e63", width=1.0,
                            linestyle="-", ylabel="Max Drawdown",
                        ))

                panel = make_panel(eq[:n], ylabel="Equity [%]",
                                   color="#2196f3", height_ratio=0.15,
                                   addplot=eq_addplots or None)
                panel.aggregation_method = "none"
                panels.append(panel)
            else:
                if plot_drawdown:
                    peak = np.maximum.accumulate(eq[:n])
                    underwater = peak > eq[:n]
                    dd_segs_abs: list[tuple[int, np.ndarray]] = []
                    i = 0
                    while i < n:
                        if underwater[i]:
                            start = i
                            level = peak[i]
                            while i < n and underwater[i] and peak[i] == level:
                                i += 1
                            dd_segs_abs.append((start, np.full(i - start, level)))
                        else:
                            i += 1
                    if dd_segs_abs:
                        eq_addplots.append(make_segments(
                            dd_segs_abs, color="#e91e63", width=1.0,
                            linestyle="-", ylabel="Max Drawdown",
                        ))

                panel = make_panel(eq[:n], ylabel="Equity",
                                   color="#2196f3", height_ratio=0.15,
                                   addplot=eq_addplots or None)
                panel.aggregation_method = "none"
                panels.append(panel)

        # Return panel (independent of equity panel)
        if plot_return and equity_curve is not None and "Equity" in equity_curve.columns:
            eq_ret = equity_curve["Equity"].values.astype(float)
            initial = eq_ret[0] if eq_ret[0] != 0 else 1.0
            ret = (eq_ret / initial - 1) * 100
            panels.append(make_panel(ret[:n], ylabel="Return [%]",
                                     color="#ff9800", height_ratio=0.12))
            panels[-1].aggregation_method = "none"

        # P/L panel with sloped trade segments
        if plot_pl and trades is not None and not trades.empty:
            panels.append(_build_pl_panel(trades, n))

        # Strategy indicators
        color_idx = 0
        if indicators:
            # Pre-process: resolve colours and overlay flags
            resolved_indicators: list[dict[str, Any]] = []
            for ind in indicators:
                ind_data = np.asarray(ind["data"], dtype=float)
                ind_name = ind.get("name", "Indicator")
                ind_color = ind.get("color", None)
                ind_scatter = ind.get("scatter", False)
                overlay = ind.get("overlay", None)
                group = ind.get("group", None)

                if ind_color is None:
                    ind_color = _INDICATOR_COLORS[
                        color_idx % len(_INDICATOR_COLORS)
                    ]
                    color_idx += 1

                if overlay is None:
                    overlay = _is_overlay(ind_data, closes)

                resolved_indicators.append({
                    "data": ind_data,
                    "name": ind_name,
                    "color": ind_color,
                    "scatter": ind_scatter,
                    "overlay": overlay,
                    "group": group,
                })

            # Group non-overlay indicators by their group key
            panel_groups: dict[str, list[dict[str, Any]]] = {}
            panel_group_order: list[str] = []
            for ri in resolved_indicators:
                if ri["overlay"]:
                    ap_type = "scatter" if ri["scatter"] else "line"
                    all_addplots.append(make_addplot(
                        ri["data"][:n], type=ap_type, color=ri["color"],
                        ylabel=ri["name"],
                    ))
                else:
                    key = ri["group"] or ri["name"]
                    if key not in panel_groups:
                        panel_groups[key] = []
                        panel_group_order.append(key)
                    panel_groups[key].append(ri)

            # Create one subpanel per group
            for key in panel_group_order:
                group_inds = panel_groups[key]
                spec = _lookup_indicator(key)

                if spec is not None and spec.outputs and len(spec.outputs) == len(group_inds):
                    # ---- Registry: full role-based layout --------------------------------
                    # Find which output is the panel primary (role 'primary' or 'histogram').
                    primary_ri: dict[str, Any] | None = None
                    primary_ospec: OutputSpec | None = None
                    addplot_pairs: list[tuple[dict[str, Any], OutputSpec]] = []
                    for ri, ospec in zip(group_inds, spec.outputs):
                        if ospec.role in ("primary", "histogram") and primary_ri is None:
                            primary_ri = ri
                            primary_ospec = ospec
                        else:
                            addplot_pairs.append((ri, ospec))
                    if primary_ri is None:
                        # All outputs are addplot-type; promote first to primary
                        primary_ri = group_inds[0]
                        primary_ospec = spec.outputs[0]
                        addplot_pairs = list(zip(group_inds[1:], spec.outputs[1:]))

                    panel_ap: list[dict[str, Any]] = []
                    for ri, ospec in addplot_pairs:
                        ap_type = "scatter" if ospec.role == "scatter" else "line"
                        panel_ap.append(make_addplot(
                            ri["data"][:n], type=ap_type,
                            color=ospec.color or ri["color"],
                            ylabel=ospec.ylabel or ri["name"],
                            width=ospec.width,
                        ))
                    for rl in spec.ref_lines:
                        panel_ap.append(make_addplot(
                            np.full(n, rl.value), type="line",
                            color=rl.color, linestyle=rl.linestyle, width=rl.width,
                        ))

                    ptype = "histogram" if (primary_ospec and primary_ospec.role == "histogram") else "line"
                    pcol = (primary_ospec.color if primary_ospec else None) or primary_ri["color"]
                    panel = make_panel(
                        primary_ri["data"][:n],
                        ylabel=primary_ospec.ylabel if primary_ospec else primary_ri["name"],
                        color=pcol or "#1f77b4",
                        height_ratio=spec.height_ratio,
                        panel_type=ptype,
                        addplot=panel_ap or None,
                    )
                    panel.zero_centered = spec.zero_centered
                    if ptype == "histogram":
                        panel.aggregation_method = "mean"
                    if primary_ospec and primary_ospec.bar_color_mode:
                        panel.bar_color_mode = primary_ospec.bar_color_mode
                    panels.append(panel)

                elif spec is not None:
                    # ---- Registry: params-only (no output roles or count mismatch) ----
                    ref_ap: list[dict[str, Any]] = [
                        make_addplot(
                            np.full(n, rl.value), type="line",
                            color=rl.color, linestyle=rl.linestyle, width=rl.width,
                        )
                        for rl in spec.ref_lines
                    ]
                    if len(group_inds) == 1:
                        ri = group_inds[0]
                        if ri["scatter"]:
                            scatter_ap = [make_addplot(
                                ri["data"][:n], type="scatter",
                                color=ri["color"], ylabel=ri["name"],
                            )] + ref_ap
                            panel = make_panel(
                                np.full(n, np.nan), ylabel=ri["name"],
                                color=ri["color"], height_ratio=spec.height_ratio,
                                panel_type=spec.panel_type, addplot=scatter_ap,
                            )
                        else:
                            panel = make_panel(
                                ri["data"][:n], ylabel=ri["name"],
                                color=ri["color"] or "#1f77b4",
                                height_ratio=spec.height_ratio,
                                panel_type=spec.panel_type,
                                addplot=ref_ap or None,
                            )
                    else:
                        first = group_inds[0]
                        extra_ap: list[dict[str, Any]] = []
                        for ri in group_inds[1:]:
                            ap_type = "scatter" if ri["scatter"] else "line"
                            extra_ap.append(make_addplot(
                                ri["data"][:n], type=ap_type,
                                color=ri["color"], ylabel=ri["name"],
                            ))
                        extra_ap.extend(ref_ap)
                        if first["scatter"]:
                            extra_ap.insert(0, make_addplot(
                                first["data"][:n], type="scatter",
                                color=first["color"], ylabel=first["name"],
                            ))
                            panel = make_panel(
                                np.full(n, np.nan), ylabel=key,
                                color=first["color"], height_ratio=spec.height_ratio,
                                panel_type=spec.panel_type, addplot=extra_ap,
                            )
                        else:
                            panel = make_panel(
                                first["data"][:n], ylabel=key,
                                color=first["color"] or "#1f77b4",
                                height_ratio=spec.height_ratio,
                                panel_type=spec.panel_type,
                                addplot=extra_ap or None,
                            )
                    panel.zero_centered = spec.zero_centered
                    panels.append(panel)

                else:
                    # ---- No registry match: generic fallback --------------------------------
                    if len(group_inds) == 1:
                        ri = group_inds[0]
                        if ri["scatter"]:
                            panel_ap = [make_addplot(
                                ri["data"][:n], type="scatter", color=ri["color"],
                                ylabel=ri["name"],
                            )]
                            panels.append(make_panel(
                                np.full(n, np.nan), ylabel=ri["name"],
                                color=ri["color"], height_ratio=0.12,
                                addplot=panel_ap,
                            ))
                        else:
                            panels.append(make_panel(
                                ri["data"][:n], ylabel=ri["name"],
                                color=ri["color"] or "#1f77b4",
                                height_ratio=0.12,
                            ))
                    else:
                        # Multiple outputs → single panel with first as primary
                        first = group_inds[0]
                        extra_ap = []
                        for ri in group_inds[1:]:
                            ap_type = "scatter" if ri["scatter"] else "line"
                            extra_ap.append(make_addplot(
                                ri["data"][:n], type=ap_type,
                                color=ri["color"], ylabel=ri["name"],
                            ))
                        if first["scatter"]:
                            extra_ap.insert(0, make_addplot(
                                first["data"][:n], type="scatter",
                                color=first["color"], ylabel=first["name"],
                            ))
                            panels.append(make_panel(
                                np.full(n, np.nan), ylabel=key,
                                color=first["color"], height_ratio=0.12,
                                addplot=extra_ap,
                            ))
                        else:
                            panels.append(make_panel(
                                first["data"][:n], ylabel=key,
                                color=first["color"] or "#1f77b4",
                                height_ratio=0.12,
                                addplot=extra_ap or None,
                            ))

        # --- initialise base widget -------------------------------------------
        show_vol = plot_volume and volumes is not None

        super().__init__(
            parent,
            dates=dates,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            chart_type="candle",
            style=resolved_style,
            show_volume=show_vol,
            title=title,
            addplots=all_addplots,
            panels=panels if panels else None,
        )

        # Expose for programmatic access
        self._equity_curve = equity_curve
        self._trades = trades
        self._indicators_data = indicators
