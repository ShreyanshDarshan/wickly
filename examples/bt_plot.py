"""
Example: wickly.bt.plot — backtesting result visualisation.

Demonstrates ``wickly.bt.plot`` as a drop-in replacement for
``backtesting.Backtest.plot()``, including:

- OHLC candles with trade entry/exit lines (green = winners, red = losers)
- Equity sub-panel (relative return %)
- P/L histogram sub-panel
- Drawdown sub-panel
- User-supplied addplot (SMA overlay) merged alongside auto-generated ones

Two usage styles are shown:

1. **With the backtesting library** — runs a real strategy and plots its
   output.  Requires ``pip install backtesting``.

2. **Standalone / synthetic** — builds a minimal ``stats``-like Series from
   hand-crafted data so the example runs without any extra dependencies.

Run:
    python examples/bt_plot.py
    python examples/bt_plot.py --synthetic   # no backtesting install needed
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import wickly


# ---------------------------------------------------------------------------
# Shared OHLCV generator
# ---------------------------------------------------------------------------

def _generate_ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(0.1, 1.2, n))
    open_ = close + rng.normal(0, 0.4, n)
    high  = np.maximum(open_, close) + rng.uniform(0.1, 1.5, n)
    low   = np.minimum(open_, close) - rng.uniform(0.1, 1.5, n)
    vol   = rng.integers(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. Synthetic path (no backtesting library needed)
# ---------------------------------------------------------------------------

def _run_synthetic() -> None:
    """Build a fake stats Series and plot it."""
    df = _generate_ohlcv()
    n  = len(df)
    rng = np.random.default_rng(7)

    # Equity curve
    equity  = 10_000 + np.cumsum(rng.normal(10, 60, n))
    peak    = np.maximum.accumulate(equity)
    dd_pct  = (equity - peak) / peak
    equity_curve = pd.DataFrame({"Equity": equity, "DrawdownPct": dd_pct})

    # Synthetic trades
    trades_list = []
    bar = 3
    while bar < n - 15:
        entry_bar   = bar
        exit_bar    = bar + rng.integers(4, 15)
        entry_price = df["Close"].iloc[entry_bar]
        exit_price  = df["Close"].iloc[exit_bar]
        pnl         = exit_price - entry_price
        trades_list.append({
            "EntryBar":   entry_bar,
            "ExitBar":    exit_bar,
            "EntryPrice": entry_price,
            "ExitPrice":  exit_price,
            "PnL":        pnl,
            "Size":       100,
            "ReturnPct":  pnl / entry_price,
        })
        bar = exit_bar + rng.integers(2, 6)

    trades = pd.DataFrame(trades_list)

    # Minimal stats Series (same keys backtesting.py uses)
    stats = pd.Series(dtype=object)
    stats["_equity_curve"] = equity_curve
    stats["_trades"]       = trades
    stats["_strategy"]     = None

    # User-supplied addplot: 20-bar SMA overlay
    sma20 = wickly.make_addplot(
        df["Close"].rolling(20).mean(),
        color="#ff9800",
        width=1.5,
        ylabel="SMA 20",
    )

    wickly.bt.plot(
        stats,
        data=df,
        plot_equity=True,
        plot_drawdown=True,
        plot_pl=True,
        relative_equity=True,
        addplot=sma20,
        title="Synthetic Backtest Result",
    )


# ---------------------------------------------------------------------------
# 2. Real backtesting.py path
# ---------------------------------------------------------------------------

def _run_with_backtesting() -> None:
    """Run a simple SMA cross strategy and plot with wickly.bt.plot."""
    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError:
        print(
            "backtesting is not installed.  Run:\n"
            "    pip install backtesting\n"
            "or use --synthetic to run without it."
        )
        sys.exit(1)

    df = _generate_ohlcv()

    # Simple dual-SMA crossover strategy
    class SmaCross(Strategy):
        fast = 10
        slow = 30

        def init(self):
            self.sma_fast = self.I(
                lambda: pd.Series(self.data.Close).rolling(self.fast).mean().values,
                name="SMA Fast",
                overlay=True,
            )
            self.sma_slow = self.I(
                lambda: pd.Series(self.data.Close).rolling(self.slow).mean().values,
                name="SMA Slow",
                overlay=True,
            )

        def next(self):
            if crossover(self.sma_fast, self.sma_slow):
                self.buy()
            elif crossover(self.sma_slow, self.sma_fast):
                self.sell()

    bt     = Backtest(df, SmaCross, cash=10_000, commission=0.002)
    stats  = bt.run()

    wickly.bt.plot(
        stats,
        plot_equity=True,
        plot_drawdown=True,
        plot_pl=True,
        relative_equity=True,
        title="SMA Cross — wickly.bt.plot",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="wickly.bt.plot example")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of running backtesting.py",
    )
    args = parser.parse_args()

    if args.synthetic:
        _run_synthetic()
    else:
        _run_with_backtesting()


if __name__ == "__main__":
    main()
