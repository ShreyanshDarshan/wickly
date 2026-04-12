"""
Wickly — Interactive candlestick chart plotting with PyQt6.

Usage mirrors mplfinance:

    import wickly
    wickly.plot(df, type='candle', volume=True, mav=(10,20), style='yahoo')
"""

from wickly._version import __version__
from wickly.plotting import plot
from wickly.addplot import make_addplot
from wickly.styles import make_style, available_styles

__all__ = [
    "__version__",
    "plot",
    "make_addplot",
    "make_style",
    "available_styles",
]
