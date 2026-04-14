"""
Example: sub-panel charts — RSI, MACD, and an equity curve.

Demonstrates ``make_panel`` with both ``panel_type='line'`` and
``panel_type='histogram'``, as well as a panel-level ``addplot`` overlay
(the RSI overbought/oversold threshold lines).

Run:
    python examples/subpanels.py
"""

import numpy as np
import pandas as pd

import wickly


def _generate_sample_data(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.2, n))
    open_ = close + rng.normal(0, 0.4, n)
    high  = np.maximum(open_, close) + rng.uniform(0.2, 1.5, n)
    low   = np.minimum(open_, close) - rng.uniform(0.2, 1.5, n)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _rsi(close: pd.Series, period: int = 14) -> np.ndarray:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi.to_numpy()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) as numpy arrays."""
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = (ema_fast - ema_slow).to_numpy()
    sig_line   = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().to_numpy()
    histogram  = macd_line - sig_line
    return macd_line, sig_line, histogram


def _equity_curve(close: pd.Series) -> np.ndarray:
    """Simulated equity: buy-and-hold starting at 100."""
    returns = close.pct_change().fillna(0).to_numpy()
    return 100.0 * np.cumprod(1 + returns)


def main() -> None:
    df = _generate_sample_data()

    # ------------------------------------------------------------------
    # 1. RSI panel (line) with overbought / oversold threshold overlays
    # ------------------------------------------------------------------
    rsi = _rsi(df["Close"])
    ob_line = np.full_like(rsi, 70.0)   # overbought threshold
    os_line = np.full_like(rsi, 30.0)   # oversold threshold

    rsi_panel = wickly.make_panel(
        rsi,
        ylabel="RSI (14)",
        height_ratio=0.18,
        color="#5c6bc0",
        panel_type="line",
        width=1.5,
        addplot=[
            wickly.make_addplot(ob_line, color="#ef5350", width=1.0, linestyle="--"),
            wickly.make_addplot(os_line, color="#26a69a", width=1.0, linestyle="--"),
        ],
    )

    # ------------------------------------------------------------------
    # 2. MACD histogram panel with signal-line overlay
    # ------------------------------------------------------------------
    macd_line, sig_line, macd_hist = _macd(df["Close"])

    macd_panel = wickly.make_panel(
        macd_hist,
        ylabel="MACD",
        height_ratio=0.18,
        color="#26a69a",
        panel_type="histogram",
        addplot=[
            wickly.make_addplot(macd_line, color="#ef5350", width=1.2),
            wickly.make_addplot(sig_line,  color="#ff9800", width=1.2, linestyle="--"),
        ],
    )

    # ------------------------------------------------------------------
    # 3. Equity-curve panel (line)
    # ------------------------------------------------------------------
    equity = _equity_curve(df["Close"])

    equity_panel = wickly.make_panel(
        equity,
        ylabel="Equity",
        height_ratio=0.18,
        color="#ffa726",
        panel_type="line",
        width=1.5,
    )

    # ------------------------------------------------------------------
    # Plot — main chart with volume bar + three sub-panels beneath
    # ------------------------------------------------------------------
    wickly.plot(
        df,
        type="candle",
        volume=True,
        mav=(20,),
        style="nightclouds",
        title="RSI  |  MACD  |  Equity",
        panels=[rsi_panel, macd_panel, equity_panel],
    )


if __name__ == "__main__":
    main()
