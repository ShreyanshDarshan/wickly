"""
Example: using the interactive indicator menu.

Run:
    python examples/indicator_menu.py

Click the **ƒx** button in the top-right corner of the chart to open
the indicator search dialog.  From there you can search for indicators
(e.g. "RSI", "MACD", "Bollinger"), configure parameters, and add them.
Overlay indicators (SMA, EMA, Bollinger Bands) render on the main chart;
panel indicators (RSI, MACD, Stochastic, …) appear as sub-panels below.

You can also add indicators programmatically via ``widget.add_indicator()``.
"""

import numpy as np
import pandas as pd

import wickly


def _generate_sample_data(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0, 1.2, n))
    open_ = close + rng.normal(0, 0.5, n)
    high  = np.maximum(open_, close) + rng.uniform(0.2, 1.5, n)
    low   = np.minimum(open_, close) - rng.uniform(0.2, 1.5, n)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def main() -> None:
    df = _generate_sample_data()

    # Open the chart (non-blocking so we can interact programmatically).
    widget, axes = wickly.plot(
        df,
        type="candle",
        volume=True,
        title="Indicator Menu Demo – click ƒx",
        style="nightclouds",
        returnfig=True,
        block=False,
    )

    # Programmatically add a couple of indicators to start with:
    widget.add_indicator("EMA", period=20)
    widget.add_indicator("RSI", period=14)

    # Show the chart (blocking).  The user can click ƒx to add more.
    widget.show()
    from PyQt6.QtWidgets import QApplication
    QApplication.instance().exec()


if __name__ == "__main__":
    main()
