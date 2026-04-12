"""
Example: multiple chart types and addplot overlays.

Run:
    python examples/addplot_overlay.py
"""

import numpy as np
import pandas as pd

import wickly


def _generate_sample_data(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-06-01", periods=n)
    close = 50.0 + np.cumsum(rng.normal(0.05, 0.8, n))
    open_ = close + rng.normal(0, 0.3, n)
    high  = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low   = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    volume = rng.integers(100_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def main() -> None:
    df = _generate_sample_data()

    # Compute a simple upper/lower Bollinger Band
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    # Build scatter signal: mark where close crossed below lower band
    signal = np.where(df["Close"].values < lower.values, df["Low"].values - 0.5, np.nan)

    apds = [
        wickly.make_addplot(upper, type="line", color="#9c27b0", width=1.2, linestyle="--"),
        wickly.make_addplot(lower, type="line", color="#9c27b0", width=1.2, linestyle="--"),
        wickly.make_addplot(signal, type="scatter", color="#ff9800", markersize=90, marker="^"),
    ]

    wickly.plot(
        df,
        type="candle",
        volume=True,
        mav=(10,),
        style="nightclouds",
        title="Bollinger Bands + Buy Signals",
        addplot=apds,
    )


if __name__ == "__main__":
    main()
