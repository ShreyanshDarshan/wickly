"""
Example: basic candlestick chart with wickly.

Run:
    python examples/basic_candle.py
"""

import numpy as np
import pandas as pd

import wickly


def _generate_sample_data(n: int = 120) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demonstration."""
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

    # --- identical call style to mplfinance ---
    wickly.plot(
        df,
        type="candle",
        volume=True,
        mav=(10, 20),
        style="yahoo",
        title="wickly — Interactive Candlestick Chart",
    )


if __name__ == "__main__":
    main()
