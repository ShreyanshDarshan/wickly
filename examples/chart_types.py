"""
Example: all chart types side-by-side (run one at a time).

Run:
    python examples/chart_types.py
"""

import numpy as np
import pandas as pd

import wickly


def _data(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2025-03-01", periods=n)
    close = 200 + np.cumsum(rng.normal(0, 2, n))
    open_ = close + rng.normal(0, 0.8, n)
    high  = np.maximum(open_, close) + rng.uniform(0.3, 2, n)
    low   = np.minimum(open_, close) - rng.uniform(0.3, 2, n)
    vol   = rng.integers(1_000_000, 8_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def main() -> None:
    df = _data()

    for chart_type in ("candle", "ohlc", "line", "hollow"):
        wickly.plot(
            df,
            type=chart_type,
            volume=True,
            mav=(5, 20),
            style="charles",
            title=f"Chart type: {chart_type}",
        )


if __name__ == "__main__":
    main()
