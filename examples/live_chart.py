"""
Example: live / animated candlestick chart with a live addplot overlay.

A new bar is appended every 2 seconds. The most recent bar is updated
with simulated live ticks between bar boundaries.  A rolling SMA overlay
is kept in sync with the price data in real time.

Run:
    python examples/live_chart.py
"""

import sys

import numpy as np
import pandas as pd
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

import wickly


SMA_PERIOD = 10


def _generate_initial_data(n: int = 60) -> pd.DataFrame:
    """Generate synthetic OHLCV data for the initial chart."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=n, freq="min")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.8, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.8, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def main() -> None:
    # Create the QApplication *before* live_plot so we hold a strong
    # reference throughout the lifetime of the script.
    app = QApplication.instance() or QApplication(sys.argv)

    df = _generate_initial_data()

    # Compute initial SMA overlay
    sma = df["Close"].rolling(SMA_PERIOD).mean()
    ap = wickly.make_addplot(sma, color="#e91e63", width=2, ylabel=f"SMA {SMA_PERIOD}")

    # Open a live chart with the SMA overlay
    widget, axes = wickly.live_plot(
        df, type="candle", volume=True, mav=(20,),
        addplot=ap,
        title="Live Chart with Addplot Overlay", style="yahoo",
    )

    rng = np.random.default_rng(99)
    last_close = df["Close"].iloc[-1]
    last_date = df.index[-1]

    def _current_sma() -> float:
        """Compute the SMA over the last SMA_PERIOD closes."""
        closes = widget._closes
        if len(closes) < SMA_PERIOD:
            return float("nan")
        return float(np.mean(closes[-SMA_PERIOD:]))

    def on_tick():
        """Called every 200ms — simulates a live tick on the current bar."""
        nonlocal last_close
        last_close += rng.normal(0, 0.3)
        widget.update_last(
            close=last_close,
            high=max(widget._highs[-1], last_close),
            low=min(widget._lows[-1], last_close),
        )
        # Keep the SMA overlay in sync with the latest close
        widget.update_addplot_last(0, _current_sma())

    def on_new_bar():
        """Called every 2s — appends a complete new bar."""
        nonlocal last_close, last_date
        last_date += pd.Timedelta(minutes=1)
        open_ = last_close + rng.normal(0, 0.1)
        close = open_ + rng.normal(0, 0.5)
        high = max(open_, close) + rng.uniform(0.05, 0.4)
        low = min(open_, close) - rng.uniform(0.05, 0.4)
        vol = float(rng.integers(100_000, 1_000_000))

        widget.append_data(
            dates=pd.DatetimeIndex([last_date]),
            opens=np.array([open_]),
            highs=np.array([high]),
            lows=np.array([low]),
            closes=np.array([close]),
            volumes=np.array([vol]),
        )
        # Append the new SMA point for the new bar
        widget.append_addplot_data(0, _current_sma())
        last_close = close

    # Tick timer — fast updates within the current candle
    tick_timer = QTimer()
    tick_timer.timeout.connect(on_tick)
    tick_timer.start(200)

    # Bar timer — new candle every 2 seconds
    bar_timer = QTimer()
    bar_timer.timeout.connect(on_new_bar)
    bar_timer.start(2000)

    app.exec()


if __name__ == "__main__":
    main()
