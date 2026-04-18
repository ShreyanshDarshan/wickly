"""Indicator registry — dataclasses and lookup functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ParamSpec:
    """Describes one tunable parameter for an indicator."""

    name: str
    label: str
    default: int | float
    min_val: int | float = 1
    max_val: int | float = 500
    step: int | float = 1


@dataclass
class OutputSpec:
    """Describes one output series produced by an indicator."""

    key: str
    label: str
    color: str = "#1f77b4"
    plot_type: str = "line"       # 'line' | 'histogram' | 'scatter'
    width: float = 1.5
    linestyle: str = "-"


@dataclass
class IndicatorSpec:
    """Full specification for a registered indicator.

    Parameters
    ----------
    name : str
        Unique identifier (e.g. ``"SMA"``).
    display_name : str
        Human-readable name shown in the indicator menu.
    category : str
        Grouping category (e.g. ``"Trend"``, ``"Momentum"``).
    overlay : bool
        ``True`` → drawn on main chart; ``False`` → own sub-panel.
    compute : callable
        ``(closes, opens, highs, lows, volumes, **params) → dict[str, ndarray]``
    params : list[ParamSpec]
        Tunable parameters.
    outputs : list[OutputSpec]
        Output series descriptions (keys must match ``compute`` return dict).
    ref_lines : list[float]
        Horizontal reference lines (e.g. RSI 30/70).
    zero_centered : bool
        If ``True`` the panel Y-range is symmetric around zero.
    height_ratio : float
        Sub-panel height fraction (ignored for overlays).
    """

    name: str
    display_name: str
    category: str
    overlay: bool
    compute: Callable[..., dict[str, np.ndarray]]
    params: list[ParamSpec] = field(default_factory=list)
    outputs: list[OutputSpec] = field(default_factory=list)
    ref_lines: list[float] = field(default_factory=list)
    zero_centered: bool = False
    height_ratio: float = 0.18


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, IndicatorSpec] = {}


def register_indicator(spec: IndicatorSpec) -> None:
    """Add an indicator to the global registry.

    If *spec.name* already exists it is silently replaced, which allows
    users to override built-in implementations.
    """
    _REGISTRY[spec.name] = spec


def get_indicator(name: str) -> IndicatorSpec:
    """Return the :class:`IndicatorSpec` for *name*, or raise ``KeyError``."""
    return _REGISTRY[name]


def list_indicators(category: str | None = None) -> list[IndicatorSpec]:
    """Return all registered indicators, optionally filtered by *category*."""
    specs = list(_REGISTRY.values())
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return sorted(specs, key=lambda s: s.display_name)


def categories() -> list[str]:
    """Return a sorted list of distinct category names."""
    return sorted({s.category for s in _REGISTRY.values()})
