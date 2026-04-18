"""Indicator registry — discover, compute and render technical indicators.

Public API
----------
- :func:`register_indicator` — add a custom indicator to the menu
- :func:`get_indicator` — look up an indicator by name
- :func:`list_indicators` — list all (or category-filtered) indicators
- :func:`categories` — list distinct category names
- :class:`IndicatorSpec`, :class:`ParamSpec`, :class:`OutputSpec` — dataclasses
"""

from wickly.indicators._registry import (
    IndicatorSpec,
    OutputSpec,
    ParamSpec,
    categories,
    get_indicator,
    list_indicators,
    register_indicator,
)

# Import builtins so they self-register at first access.
import wickly.indicators._builtins as _builtins  # noqa: F401

__all__ = [
    "IndicatorSpec",
    "OutputSpec",
    "ParamSpec",
    "categories",
    "get_indicator",
    "list_indicators",
    "register_indicator",
]
