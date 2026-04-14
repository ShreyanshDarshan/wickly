"""Tests for wickly.bt.plot() — backtesting integration (non-GUI smoke tests)."""

import numpy as np
import pandas as pd
import pytest

import wickly
import wickly.bt
from wickly.bt.chart_widget import BacktestWidget, _is_overlay, _build_pl_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df(n: int = 60) -> pd.DataFrame:
    """Synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.3, n),
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1000, 9999, n).astype(float),
        },
        index=dates,
    )


def _sample_equity(n: int = 60) -> pd.DataFrame:
    """Synthetic equity curve with Equity and DrawdownPct columns."""
    rng = np.random.default_rng(7)
    equity = 10_000 + np.cumsum(rng.normal(5, 20, n))
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak
    return pd.DataFrame({"Equity": equity, "DrawdownPct": dd_pct})


def _sample_trades(n: int = 60, num_trades: int = 5) -> pd.DataFrame:
    """Synthetic trade log matching backtesting.py layout."""
    rng = np.random.default_rng(3)
    records = []
    bar = 2
    for _ in range(num_trades):
        entry_bar = bar
        exit_bar = min(bar + rng.integers(3, 10), n - 1)
        entry_price = 100 + rng.normal(0, 2)
        exit_price = entry_price + rng.normal(1, 3)
        pnl = exit_price - entry_price
        records.append({
            "EntryBar": entry_bar,
            "ExitBar": exit_bar,
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "PnL": pnl,
            "Size": 100,
            "ReturnPct": pnl / entry_price,
        })
        bar = exit_bar + 2
        if bar >= n - 3:
            break
    return pd.DataFrame(records)


def _mock_stats(
    n: int = 60,
    num_trades: int = 5,
    include_strategy: bool = False,
) -> pd.Series:
    """Build a minimal stats Series similar to backtesting.Backtest.run()."""
    data = _sample_df(n)
    equity = _sample_equity(n)
    trades = _sample_trades(n, num_trades)
    s = pd.Series(dtype=object)
    s["_equity_curve"] = equity
    s["_trades"] = trades
    s["_strategy"] = None
    return s, data


def _safe_close(widget) -> None:
    try:
        from PyQt6 import sip
        if not sip.isdeleted(widget):
            widget.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests — BacktestWidget directly
# ---------------------------------------------------------------------------

class TestBacktestWidget:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_basic_construction(self):
        stats, data = _mock_stats()
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            trades=stats["_trades"],
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_no_equity(self):
        stats, data = _mock_stats()
        w = BacktestWidget(data=data, trades=stats["_trades"],
                           plot_equity=False)
        # Should have no equity panel
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_no_trades(self):
        stats, data = _mock_stats()
        w = BacktestWidget(data=data, equity_curve=stats["_equity_curve"],
                           trades=pd.DataFrame(), plot_trades=True)
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_drawdown_panel(self):
        stats, data = _mock_stats()
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            plot_drawdown=True,
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_absolute_equity(self):
        stats, data = _mock_stats()
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            relative_equity=False,
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_smooth_equity(self):
        stats, data = _mock_stats()
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            smooth_equity=True,
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_with_user_addplot(self):
        stats, data = _mock_stats()
        ap = wickly.make_addplot(
            data["Close"].rolling(5).mean(), color="blue", ylabel="SMA5"
        )
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            trades=stats["_trades"],
            addplots=[ap],
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_with_indicator_overlay(self):
        stats, data = _mock_stats()
        ind = {
            "data": data["Close"].rolling(10).mean().values,
            "name": "SMA10",
            "overlay": True,
            "color": "#ff0000",
            "scatter": False,
        }
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            indicators=[ind],
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_with_indicator_panel(self):
        stats, data = _mock_stats()
        rsi = np.random.default_rng(0).uniform(20, 80, len(data))
        ind = {
            "data": rsi,
            "name": "RSI",
            "overlay": False,
            "color": "#9c27b0",
        }
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            indicators=[ind],
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_with_indicator_auto_classify(self):
        stats, data = _mock_stats()
        # Near-close values should be auto-classified as overlay
        ind = {
            "data": data["Close"].rolling(5).mean().values,
            "name": "SMA5",
        }
        w = BacktestWidget(
            data=data,
            equity_curve=stats["_equity_curve"],
            indicators=[ind],
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)

    def test_all_panels_off(self):
        stats, data = _mock_stats()
        w = BacktestWidget(
            data=data,
            plot_equity=False,
            plot_volume=False,
            plot_trades=False,
            plot_pl=False,
        )
        assert isinstance(w, BacktestWidget)
        _safe_close(w)


# ---------------------------------------------------------------------------
# Tests — wickly.bt.plot() orchestration
# ---------------------------------------------------------------------------

class TestBtPlot:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_returnfig_basic(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(stats, data=data, returnfig=True)
        assert result is not None
        widget, axes = result
        assert isinstance(widget, BacktestWidget)
        assert "main" in axes
        _safe_close(widget)

    def test_returnfig_no_equity(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(stats, data=data, plot_equity=False,
                                returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_with_drawdown(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(stats, data=data, plot_drawdown=True,
                                returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_empty_trades(self):
        stats, data = _mock_stats(num_trades=0)
        result = wickly.bt.plot(stats, data=data, returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_with_addplot(self):
        stats, data = _mock_stats()
        ap = wickly.make_addplot(
            data["Close"].rolling(5).mean(), color="orange",
        )
        result = wickly.bt.plot(stats, data=data, addplot=ap, returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_single_addplot_dict(self):
        stats, data = _mock_stats()
        ap = wickly.make_addplot(data["Close"].rolling(3).mean())
        result = wickly.bt.plot(stats, data=data, addplot=ap, returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_style(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(stats, data=data, style="yahoo",
                                returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_figsize(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(stats, data=data, figsize=(800, 400),
                                returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_no_data_raises(self):
        stats, _ = _mock_stats()
        with pytest.raises(ValueError, match="No OHLCV data"):
            wickly.bt.plot(stats, returnfig=True)

    def test_savefig(self, tmp_path):
        stats, data = _mock_stats()
        path = str(tmp_path / "chart.png")
        result = wickly.bt.plot(stats, data=data, savefig=path,
                                returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_return_panel(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(stats, data=data, plot_return=True,
                                returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_all_panels_enabled(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(
            stats, data=data,
            plot_equity=True, plot_return=True, plot_drawdown=True,
            plot_volume=True, plot_trades=True, plot_pl=True,
            returnfig=True,
        )
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_smooth_and_relative(self):
        stats, data = _mock_stats()
        result = wickly.bt.plot(
            stats, data=data,
            smooth_equity=True, relative_equity=True,
            returnfig=True,
        )
        assert result is not None
        widget, _ = result
        _safe_close(widget)


# ---------------------------------------------------------------------------
# Tests — helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_is_overlay_true(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        values = np.array([100.5, 101.2, 102.1, 103.3, 104.2])
        assert _is_overlay(values, closes) is True

    def test_is_overlay_false(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        values = np.array([30.0, 45.0, 55.0, 70.0, 80.0])
        assert _is_overlay(values, closes) is False

    def test_is_overlay_with_nans(self):
        closes = np.array([100.0, np.nan, 102.0, 103.0, 104.0])
        values = np.array([100.5, 101.2, np.nan, 103.3, 104.2])
        assert _is_overlay(values, closes) is True

    def test_is_overlay_insufficient_data(self):
        closes = np.array([np.nan])
        values = np.array([np.nan])
        assert _is_overlay(values, closes) is False

    def test_build_pl_data_basic(self):
        trades = pd.DataFrame({
            "ExitBar": [5, 10],
            "PnL": [100.0, -50.0],
        })
        pl = _build_pl_data(trades, 15)
        assert pl[5] == 100.0
        assert pl[10] == -50.0
        assert np.isnan(pl[0])
        assert np.isnan(pl[7])

    def test_build_pl_data_empty(self):
        pl = _build_pl_data(pd.DataFrame(), 10)
        assert np.all(np.isnan(pl))

    def test_bt_accessible_from_wickly(self):
        """wickly.bt.plot should be accessible after `import wickly`."""
        assert hasattr(wickly, "bt")
        assert hasattr(wickly.bt, "plot")
        assert hasattr(wickly.bt, "BacktestWidget")
