"""wickly.bt — Backtesting chart integration.

Drop-in replacement for ``backtesting.Backtest.plot()`` using native
PyQt6 rendering:

    import wickly.bt
    wickly.bt.plot(stats)
"""

from wickly.bt.chart_widget import BacktestWidget
from wickly.bt.plotting import plot

__all__ = ["BacktestWidget", "plot"]
