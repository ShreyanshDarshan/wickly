"""Known TA-Lib indicator panel specifications.

Each entry maps an uppercase indicator prefix to an ``IndicatorSpec``
describing how its outputs should be rendered and what reference lines
to draw.  Users can add their own entries via :func:`register_indicator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OutputSpec:
    """Rendering spec for one output of a (potentially multi-output) indicator.

    Parameters
    ----------
    role : str
        ``'primary'``   — this output provides the main panel data (line type).
        ``'histogram'`` — this output provides the main panel data (bar type).
        ``'line'``      — rendered as an addplot overlay line.
        ``'scatter'``   — rendered as an addplot scatter.
    color : str or None
        Line / bar colour.  ``None`` inherits the resolved indicator colour.
    ylabel : str or None
        Label override for the output.  ``None`` uses the indicator name.
    width : float
        Line / bar width.
    bar_color_mode : str or None
        When ``role='histogram'``, controls per-bar dynamic colouring.
        ``'macd'`` — green (lighter when descending) for positive bars,
        red (lighter when ascending) for negative bars.
    """

    role: str = "line"          # 'primary' | 'histogram' | 'line' | 'scatter'
    color: str | None = None
    ylabel: str | None = None
    width: float = 1.5
    bar_color_mode: str | None = None  # None | 'macd'


@dataclass
class RefLine:
    """Horizontal reference line drawn across a sub-panel.

    Parameters
    ----------
    value : float
        Y-axis value for the horizontal line.
    color : str
        Line colour.
    linestyle : str
        Matplotlib linestyle string (``'--'``, ``':'``, ``'-'``, …).
    width : float
        Line width.
    """

    value: float
    color: str = "#888888"
    linestyle: str = "--"
    width: float = 0.8


@dataclass
class IndicatorSpec:
    """Full rendering specification for a group of indicator outputs.

    Parameters
    ----------
    outputs : list of OutputSpec
        One entry per output **in talib return order**.  If empty the generic
        rendering logic is used for output routing, but ``ref_lines``,
        ``zero_centered``, and ``height_ratio`` are still applied.
    ref_lines : list of RefLine
        Horizontal reference lines to overlay on the panel.
    panel_type : str
        Fallback panel type used when ``outputs`` is empty and the group
        has exactly one element.  Either ``'line'`` or ``'histogram'``.
    zero_centered : bool
        If ``True`` the panel Y-axis is kept symmetric around zero.
    height_ratio : float
        Panel height relative to the main candle chart.
    """

    outputs: list[OutputSpec] = field(default_factory=list)
    ref_lines: list[RefLine] = field(default_factory=list)
    panel_type: str = "line"
    zero_centered: bool = False
    height_ratio: float = 0.12


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY: dict[str, IndicatorSpec] = {}


def lookup(group_key: str) -> IndicatorSpec | None:
    """Return the ``IndicatorSpec`` for *group_key*, or ``None``.

    Matching is done by **uppercase prefix**: the registry key must be a
    prefix of the uppercased *group_key* (e.g. ``"RSI"`` matches
    ``"RSI(close, 14)"`` and ``"RSI(close, 20)"``).
    """
    upper = group_key.upper()
    for prefix, spec in INDICATOR_REGISTRY.items():
        if (
            upper == prefix
            or upper.startswith(prefix + "(")
            or upper.startswith(prefix + "_")
        ):
            return spec
    return None


def register_indicator(prefix: str, spec: IndicatorSpec) -> None:
    """Register a custom indicator spec.

    Parameters
    ----------
    prefix : str
        Uppercase prefix that identifies the indicator family
        (e.g. ``"RSI"``, ``"MYIND"``).  Matching is case-insensitive.
    spec : IndicatorSpec
        The rendering specification to apply when this prefix is matched.

    Examples
    --------
    >>> from wickly.bt._indicator_registry import IndicatorSpec, OutputSpec, RefLine
    >>> register_indicator("MYOSC", IndicatorSpec(
    ...     ref_lines=[RefLine(0), RefLine(50, color="#ff0000")],
    ...     zero_centered=True,
    ... ))
    """
    INDICATOR_REGISTRY[prefix.upper()] = spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref(
    *values: float,
    color: str = "#888888",
    linestyle: str = "--",
    width: float = 0.8,
) -> list[RefLine]:
    """Build a list of ``RefLine`` objects at the given y-values."""
    return [RefLine(v, color=color, linestyle=linestyle, width=width) for v in values]


# ---------------------------------------------------------------------------
# Built-in entries
# ---------------------------------------------------------------------------

# --- Oscillators: 0–100 range ---
register_indicator("RSI",    IndicatorSpec(ref_lines=_ref(30, 70)))
register_indicator("MFI",    IndicatorSpec(ref_lines=_ref(20, 80)))
register_indicator("ULTOSC", IndicatorSpec(ref_lines=_ref(30, 70)))

# --- Stochastics: 0–100 range, two outputs (K, D) ---
_STOCH_OUTPUTS = [
    OutputSpec(role="primary", color="#1f77b4", ylabel="K"),
    OutputSpec(role="line",    color="#ff7f0e", ylabel="D"),
]
register_indicator("STOCH",    IndicatorSpec(outputs=list(_STOCH_OUTPUTS), ref_lines=_ref(20, 80)))
register_indicator("STOCHRSI", IndicatorSpec(outputs=list(_STOCH_OUTPUTS), ref_lines=_ref(20, 80)))
register_indicator("STOCHF",   IndicatorSpec(outputs=list(_STOCH_OUTPUTS), ref_lines=_ref(20, 80)))

# --- Zero-centred oscillators (single output) ---
register_indicator("CCI",      IndicatorSpec(ref_lines=_ref(-100, 100), zero_centered=True))
register_indicator("CMO",      IndicatorSpec(ref_lines=_ref(-50, 50),   zero_centered=True))
register_indicator("WILLR",    IndicatorSpec(ref_lines=_ref(-20, -80)))
register_indicator("AROONOSC", IndicatorSpec(ref_lines=_ref(0),         zero_centered=True))
register_indicator("PPO",      IndicatorSpec(ref_lines=_ref(0),         zero_centered=True))
register_indicator("APO",      IndicatorSpec(ref_lines=_ref(0),         zero_centered=True))
register_indicator("TRIX",     IndicatorSpec(ref_lines=_ref(0),         zero_centered=True))
register_indicator("ROC",      IndicatorSpec(ref_lines=_ref(0),         zero_centered=True))
register_indicator("MOM",      IndicatorSpec(ref_lines=_ref(0),         zero_centered=True))

# --- AROON: two outputs (AroonDown, AroonUp) ---
register_indicator("AROON", IndicatorSpec(
    outputs=[
        OutputSpec(role="primary", color="#ef5350", ylabel="Down"),
        OutputSpec(role="line",    color="#26a69a", ylabel="Up"),
    ],
    ref_lines=_ref(30, 70),
))

# --- ADX family: single output, reference line at 25 ---
register_indicator("ADX",  IndicatorSpec(ref_lines=_ref(25)))
register_indicator("ADXR", IndicatorSpec(ref_lines=_ref(25)))
register_indicator("DX",   IndicatorSpec(ref_lines=_ref(25)))

# --- MACD family: talib order → [macd_line, signal, histogram] ---
_MACD_OUTPUTS = [
    OutputSpec(role="line",      color="#1f77b4", ylabel="MACD",   width=1.2),
    OutputSpec(role="line",      color="#ff7f0e", ylabel="Signal", width=1.2),
    OutputSpec(role="histogram", color="#26a69a", ylabel="Hist",   bar_color_mode="macd"),
]
register_indicator("MACD",    IndicatorSpec(outputs=list(_MACD_OUTPUTS), ref_lines=_ref(0), zero_centered=True))
register_indicator("MACDEXT", IndicatorSpec(outputs=list(_MACD_OUTPUTS), ref_lines=_ref(0), zero_centered=True))
register_indicator("MACDFIX", IndicatorSpec(outputs=list(_MACD_OUTPUTS), ref_lines=_ref(0), zero_centered=True))

# --- Volatility / volume: no reference lines, standalone panels ---
register_indicator("ATR",  IndicatorSpec())
register_indicator("NATR", IndicatorSpec())
register_indicator("OBV",  IndicatorSpec())
