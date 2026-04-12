"""Built-in styles and style helpers (mplfinance-compatible)."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Built-in style definitions
# ---------------------------------------------------------------------------

_BUILTIN_STYLES: dict[str, dict[str, Any]] = {
    "default": {
        "up_color":    "#26a69a",
        "down_color":  "#ef5350",
        "edge_up":     "#26a69a",
        "edge_down":   "#ef5350",
        "wick_up":     "#26a69a",
        "wick_down":   "#ef5350",
        "volume_up":   "#26a69a",
        "volume_down": "#ef5350",
        "bg_color":    "#ffffff",
        "grid_color":  "#e0e0e0",
        "text_color":  "#333333",
        "alpha":       0.9,
    },
    "charles": {
        "up_color":    "#006340",
        "down_color":  "#a02128",
        "edge_up":     "#006340",
        "edge_down":   "#a02128",
        "wick_up":     "#006340",
        "wick_down":   "#a02128",
        "volume_up":   "#007a00",
        "volume_down": "#d50d18",
        "bg_color":    "#ffffff",
        "grid_color":  "#a0a0a0",
        "text_color":  "#333333",
        "alpha":       1.0,
    },
    "mike": {
        "up_color":    "#000000",
        "down_color":  "#0080ff",
        "edge_up":     "#ffffff",
        "edge_down":   "#0080ff",
        "wick_up":     "#ffffff",
        "wick_down":   "#ffffff",
        "volume_up":   "#7189aa",
        "volume_down": "#7189aa",
        "bg_color":    "#0a0a23",
        "grid_color":  "#333355",
        "text_color":  "#cccccc",
        "alpha":       1.0,
    },
    "yahoo": {
        "up_color":    "#00c853",
        "down_color":  "#ff1744",
        "edge_up":     "#00c853",
        "edge_down":   "#ff1744",
        "wick_up":     "#00c853",
        "wick_down":   "#ff1744",
        "volume_up":   "#00c853",
        "volume_down": "#ff1744",
        "bg_color":    "#ffffff",
        "grid_color":  "#e6e6e6",
        "text_color":  "#333333",
        "alpha":       1.0,
    },
    "classic": {
        "up_color":    "#ffffff",
        "down_color":  "#000000",
        "edge_up":     "#000000",
        "edge_down":   "#000000",
        "wick_up":     "#000000",
        "wick_down":   "#000000",
        "volume_up":   "#181818",
        "volume_down": "#181818",
        "bg_color":    "#ffffff",
        "grid_color":  "#cccccc",
        "text_color":  "#333333",
        "alpha":       0.9,
    },
    "nightclouds": {
        "up_color":    "#00e676",
        "down_color":  "#ff5252",
        "edge_up":     "#00e676",
        "edge_down":   "#ff5252",
        "wick_up":     "#888888",
        "wick_down":   "#888888",
        "volume_up":   "#00e676",
        "volume_down": "#ff5252",
        "bg_color":    "#1e1e2f",
        "grid_color":  "#2c2c3e",
        "text_color":  "#d4d4d4",
        "alpha":       0.95,
    },
}

# Colours used for moving-average lines per style
_MAV_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


def available_styles() -> list[str]:
    """Return a list of available built-in style names."""
    return list(_BUILTIN_STYLES.keys())


def _get_style(name_or_dict: str | dict | None) -> dict[str, Any]:
    """Resolve a style name or dict into a full style dictionary."""
    if name_or_dict is None:
        return _BUILTIN_STYLES["default"].copy()
    if isinstance(name_or_dict, dict):
        base = _BUILTIN_STYLES["default"].copy()
        base.update(name_or_dict)
        return base
    name = name_or_dict.lower()
    if name not in _BUILTIN_STYLES:
        raise ValueError(
            f"Unknown style '{name_or_dict}'. "
            f"Available: {available_styles()}"
        )
    return _BUILTIN_STYLES[name].copy()


def make_style(
    *,
    base_mpf_style: str = "default",
    up_color: str | None = None,
    down_color: str | None = None,
    edge_up: str | None = None,
    edge_down: str | None = None,
    wick_up: str | None = None,
    wick_down: str | None = None,
    volume_up: str | None = None,
    volume_down: str | None = None,
    bg_color: str | None = None,
    grid_color: str | None = None,
    text_color: str | None = None,
    alpha: float | None = None,
    mavcolors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a style dict — similar to ``mplfinance.make_mpf_style``."""
    style = _get_style(base_mpf_style)
    overrides: dict[str, Any] = {
        k: v
        for k, v in dict(
            up_color=up_color,
            down_color=down_color,
            edge_up=edge_up,
            edge_down=edge_down,
            wick_up=wick_up,
            wick_down=wick_down,
            volume_up=volume_up,
            volume_down=volume_down,
            bg_color=bg_color,
            grid_color=grid_color,
            text_color=text_color,
            alpha=alpha,
        ).items()
        if v is not None
    }
    style.update(overrides)
    if mavcolors is not None:
        style["mavcolors"] = mavcolors
    return style
