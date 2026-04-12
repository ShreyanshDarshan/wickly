"""
Wickly — Interactive candlestick chart plotting with PyQt6.

Usage mirrors mplfinance:

    import wickly
    wickly.plot(df, type='candle', volume=True, mav=(10,20), style='yahoo')
"""

from wickly._version import __version__
from wickly.plotting import plot, live_plot
from wickly.addplot import make_addplot, make_segments
from wickly.styles import make_style, available_styles

__all__ = [
    "__version__",
    "plot",
    "live_plot",
    "make_addplot",
    "make_segments",
    "make_style",
    "available_styles",
]
