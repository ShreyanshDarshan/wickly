"""Tests for candle-budget aggregation (level-of-detail) in CandlestickWidget."""

import math

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtGui import QPainter, QPixmap

import wickly
from wickly.chart_widget import CandlestickWidget
from wickly.addplot import make_addplot, make_panel, make_segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(1)
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


def _safe_close(widget) -> None:
    try:
        from PyQt6 import sip
        if not sip.isdeleted(widget):
            widget.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests — aggregation factor
# ---------------------------------------------------------------------------

class TestAggFactor:
    def test_below_budget_no_aggregation(self):
        df = _sample_df(200)
        result = wickly.plot(df, type="candle", returnfig=True)
        widget, _ = result
        assert widget._compute_agg_factor() == 1
        _safe_close(widget)

    def test_above_budget_triggers_aggregation(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="candle", returnfig=True)
        widget, _ = result
        factor = widget._compute_agg_factor()
        assert factor > 1
        assert factor == math.ceil(2000 / widget.CANDLE_BUDGET)
        _safe_close(widget)

    def test_exact_budget_no_aggregation(self):
        budget = CandlestickWidget.CANDLE_BUDGET
        df = _sample_df(budget)
        result = wickly.plot(df, type="candle", returnfig=True)
        widget, _ = result
        assert widget._compute_agg_factor() == 1
        _safe_close(widget)


# ---------------------------------------------------------------------------
# Tests — rendering with large datasets
# ---------------------------------------------------------------------------

class TestAggregatedRendering:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_large_candle_chart(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="candle", returnfig=True)
        assert result is not None
        widget, _ = result
        # Force a paint to exercise aggregation path
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_ohlc_chart(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="ohlc", returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_line_chart(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="line", returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_hollow_chart(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="hollow", returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_volume(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="candle", volume=True, returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_mav(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="candle", mav=(10, 50), returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_addplot_line(self):
        df = _sample_df(2000)
        ap = make_addplot(df["Close"].rolling(20).mean(), color="blue",
                          ylabel="SMA20")
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_addplot_scatter(self):
        df = _sample_df(2000)
        signal = np.where(
            df["Close"].values < df["Close"].rolling(20).mean().values,
            df["Low"].values - 0.5,
            np.nan,
        )
        ap = make_addplot(signal, type="scatter", color="red", ylabel="Sig")
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_segments(self):
        df = _sample_df(2000)
        segs = [(100, np.linspace(105, 110, 20)),
                (500, np.linspace(108, 103, 30))]
        ap = make_segments(segs, color="cyan", ylabel="Segs")
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_panels(self):
        df = _sample_df(2000)
        rsi = np.random.default_rng(0).uniform(20, 80, len(df))
        panel = make_panel(rsi, ylabel="RSI", color="#9c27b0",
                           height_ratio=0.15)
        result = wickly.plot(df, type="candle",
                             panels=[panel], returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)

    def test_large_with_histogram_panel(self):
        df = _sample_df(2000)
        hist_data = np.random.default_rng(5).normal(0, 10, len(df))
        panel = make_panel(hist_data, ylabel="Hist", panel_type="histogram",
                           height_ratio=0.12)
        result = wickly.plot(df, type="candle",
                             panels=[panel], returnfig=True)
        widget, _ = result
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()
        _safe_close(widget)


# ---------------------------------------------------------------------------
# Tests — data integrity after paint
# ---------------------------------------------------------------------------

class TestAggDataRestoration:
    """Verify original data is fully restored after an aggregated paint."""

    def test_data_restored_after_paint(self):
        df = _sample_df(2000)
        result = wickly.plot(df, type="candle", volume=True, mav=(10,),
                             returnfig=True)
        widget, _ = result

        # Snapshot original state
        orig_n = widget._n
        orig_opens = widget._opens.copy()
        orig_view_start = widget._view_start
        orig_view_end = widget._view_end

        # Paint (triggers aggregation internally)
        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()

        # Verify restoration
        assert widget._n == orig_n
        np.testing.assert_array_equal(widget._opens, orig_opens)
        assert widget._view_start == orig_view_start
        assert widget._view_end == orig_view_end
        _safe_close(widget)

    def test_zoomed_in_no_aggregation(self):
        """When zoomed in below budget, factor=1, no aggregation."""
        df = _sample_df(2000)
        result = wickly.plot(df, type="candle", returnfig=True)
        widget, _ = result

        # Simulate a zoom-in to 100 bars
        widget._view_start = 500
        widget._view_end = 600
        assert widget._compute_agg_factor() == 1

        pixmap = QPixmap(200, 150)
        painter = QPainter(pixmap)
        widget._paint(painter)
        painter.end()

        assert widget._n == 2000
        _safe_close(widget)


# ---------------------------------------------------------------------------
# Tests — static helper methods
# ---------------------------------------------------------------------------

class TestAggHelperMethods:
    def test_agg_1d_last(self):
        data = np.arange(10, dtype=float)
        result = CandlestickWidget._agg_1d(data, 5, 10, "last")
        # Groups [0-4], [5-9] → last values are 4, 9
        np.testing.assert_array_equal(result, [4.0, 9.0])

    def test_agg_1d_sum(self):
        data = np.ones(10, dtype=float)
        result = CandlestickWidget._agg_1d(data, 5, 10, "sum")
        np.testing.assert_array_equal(result, [5.0, 5.0])

    def test_agg_1d_with_nans(self):
        data = np.array([1, np.nan, 3, np.nan, 5, 6, 7, 8, 9, 10],
                        dtype=float)
        result = CandlestickWidget._agg_1d(data, 5, 10, "last")
        np.testing.assert_array_equal(result, [5.0, 10.0])

    def test_agg_1d_sum_nans_as_zero(self):
        data = np.array([1, np.nan, 3, np.nan, 5], dtype=float)
        result = CandlestickWidget._agg_1d(data, 5, 5, "sum")
        # NaN treated as 0: 1+0+3+0+5 = 9
        np.testing.assert_array_almost_equal(result, [9.0])

    def test_agg_1d_short_data_padded(self):
        data = np.array([1, 2, 3], dtype=float)
        result = CandlestickWidget._agg_1d(data, 5, 5, "last")
        # Data only has 3 values, padded with NaN to length 5
        # Group [0-4]: last = data[4] = NaN (padded)
        assert len(result) == 1

    def test_agg_1d_scatter(self):
        data = np.array([np.nan, np.nan, 5.0, np.nan, np.nan,
                         3.0, np.nan, np.nan, np.nan, np.nan],
                        dtype=float)
        result = CandlestickWidget._agg_1d_scatter(data, 5, 10)
        # Group [0-4]: first non-NaN = 5.0
        # Group [5-9]: first non-NaN = 3.0
        np.testing.assert_array_equal(result, [5.0, 3.0])

    def test_agg_1d_scatter_all_nan(self):
        data = np.full(10, np.nan)
        result = CandlestickWidget._agg_1d_scatter(data, 5, 10)
        assert np.all(np.isnan(result))

    def test_uneven_groups(self):
        data = np.arange(7, dtype=float)
        result = CandlestickWidget._agg_1d(data, 5, 7, "last")
        # Groups: [0-4] → last=4, [5-6] → last=6
        np.testing.assert_array_equal(result, [4.0, 6.0])
