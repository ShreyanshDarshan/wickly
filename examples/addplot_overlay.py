"""
Example: addplot overlays — Bollinger Bands, buy signals, and overlapping segments.

Demonstrates three addplot types:

- ``type='line'``     — Bollinger Band envelopes (continuous, NaN-gapped for warm-up).
- ``type='scatter'``  — buy-signal markers where price crosses below the lower band.
- ``type='segments'`` — Knoxville-style divergence windows rendered as independent,
                        possibly overlapping broken-line segments via ``make_segments``.

Run:
    python examples/addplot_overlay.py
"""

import numpy as np
import pandas as pd

import wickly


def _generate_sample_data(n: int = 120) -> pd.DataFrame:
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


def _find_divergence_segments(
    close: pd.Series,
    low: pd.Series,
    mom_period: int = 20,
    window: int = 6,
) -> list[tuple[int, np.ndarray]]:
    """Return a list of (start_index, y_values) tuples marking bullish divergence windows.

    Bullish divergence: price makes a new rolling low but momentum does not —
    suggesting selling pressure is weakening.  Each detection spawns an
    independent segment of ``window`` bars anchored to the low at that bar.
    Because a new signal can fire before the previous window ends, segments
    may overlap — which is exactly what ``make_segments`` is designed for.
    """
    momentum = close - close.shift(mom_period)
    price_at_low = close == close.rolling(mom_period).min()
    # Bullish: price at new low but momentum higher than it was mom_period ago
    bullish = price_at_low & (momentum > momentum.shift(mom_period))

    segments: list[tuple[int, np.ndarray]] = []
    close_arr = close.values
    low_arr = low.values
    n = len(close_arr)

    for i in bullish[bullish].index:
        bar = close.index.get_loc(i)
        end = min(bar + window, n)
        # Anchor the segment to the low at the signal bar, then follow close
        y = np.empty(end - bar)
        y[0] = low_arr[bar] - 0.3          # start slightly below the wick
        y[1:] = close_arr[bar + 1 : end]   # continue along close for visibility
        segments.append((bar, y))

    return segments


def main() -> None:
    df = _generate_sample_data()

    # ------------------------------------------------------------------ #
    # 1. Bollinger Bands  (type='line', NaN during warm-up period)        #
    # ------------------------------------------------------------------ #
    sma   = df["Close"].rolling(20).mean()
    std   = df["Close"].rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std

    # ------------------------------------------------------------------ #
    # 2. Buy-signal scatter  (type='scatter')                             #
    #    Mark bars where close crosses below the lower band.              #
    # ------------------------------------------------------------------ #
    signal = np.where(
        df["Close"].values < lower.values,
        df["Low"].values - 0.5,
        np.nan,
    )

    # ------------------------------------------------------------------ #
    # 3. Divergence segments  (type='segments' via make_segments)         #
    #    Each divergence window is an independent segment so overlapping  #
    #    signals render as separate lines rather than one merged blob.    #
    # ------------------------------------------------------------------ #
    div_segs = _find_divergence_segments(df["Close"], df["Low"])
    divergence_apd = wickly.make_segments(
        div_segs,
        color="#00bcd4",
        width=2.0,
        linestyle="-",
        ylabel="Divergence",
    )

    apds = [
        # Bollinger Band envelopes
        wickly.make_addplot(upper, type="line", color="#9c27b0", width=1.2, linestyle="--",
                            ylabel="Upper BB"),
        wickly.make_addplot(lower, type="line", color="#9c27b0", width=1.2, linestyle="--",
                            ylabel="Lower BB"),
        # Scatter buy signals
        wickly.make_addplot(signal, type="scatter", color="#ff9800", markersize=90, marker="^",
                            ylabel="Buy Signal"),
        # Overlapping divergence segments
        divergence_apd,
    ]

    wickly.plot(
        df,
        type="candle",
        volume=True,
        mav=(10,),
        style="nightclouds",
        title="Bollinger Bands + Buy Signals + Divergence Segments",
        addplot=apds,
    )


if __name__ == "__main__":
    main()

