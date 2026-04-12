"""Tests for the top-level wickly.plot() function (non-GUI smoke tests)."""

import numpy as np
import pandas as pd
import pytest

import wickly
from wickly.plotting import _VALID_TYPES
from wickly.chart_widget import CandlestickWidget
from wickly.chart_widget import CandlestickWidget


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
    """Close a QWidget only if it has not already been deleted by Qt."""
    try:
        from PyQt6 import sip
        if not sip.isdeleted(widget):
            widget.close()
    except Exception:
        pass


class TestPlotValidation:
    def test_invalid_type_raises(self):
        df = _sample_df()
        with pytest.raises(ValueError, match="Invalid chart type"):
            wickly.plot(df, type="renko", returnfig=True)

    def test_volume_without_data_raises(self):
        df = _sample_df().drop(columns=["Volume"])
        with pytest.raises(ValueError, match="volume=True but no Volume"):
            wickly.plot(df, volume=True, returnfig=True)


class TestPlotReturnfig:
    """These tests use returnfig=True so they don't block."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_returnfig_candle(self):
        df = _sample_df()
        result = wickly.plot(df, type="candle", returnfig=True)
        assert result is not None
        widget, axes = result
        _safe_close(widget)

    def test_returnfig_ohlc(self):
        df = _sample_df()
        result = wickly.plot(df, type="ohlc", returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_line(self):
        df = _sample_df()
        result = wickly.plot(df, type="line", returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_with_volume(self):
        df = _sample_df()
        result = wickly.plot(df, type="candle", volume=True, returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_with_mav(self):
        df = _sample_df()
        result = wickly.plot(df, type="candle", mav=(5, 10), returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_returnfig_with_addplot(self):
        df = _sample_df()
        ap = wickly.make_addplot(df["Close"].rolling(5).mean(), color="blue")
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        assert result is not None
        widget, _ = result
        _safe_close(widget)

    def test_savefig(self, tmp_path):
        df = _sample_df()
        path = str(tmp_path / "chart.png")
        result = wickly.plot(df, type="candle", savefig=path, returnfig=True)
        assert result is not None
        widget, _ = result
        import os
        assert os.path.exists(path)
        _safe_close(widget)

    def test_legend_with_mav(self):
        """MAVs should automatically appear in the legend."""
        df = _sample_df()
        result = wickly.plot(df, type="candle", mav=(5, 10), returnfig=True)
        assert result is not None
        widget, _ = result
        # Widget should have stored MAV data for both periods
        assert len(widget._mavs) == 2
        _safe_close(widget)

    def test_legend_with_labelled_addplot(self):
        """Addplots with ylabel= should appear in the legend."""
        df = _sample_df()
        ap = wickly.make_addplot(
            df["Close"].rolling(5).mean(), color="blue", ylabel="SMA 5",
        )
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        assert result is not None
        widget, _ = result
        assert widget._addplots[0]["ylabel"] == "SMA 5"
        _safe_close(widget)

    def test_legend_addplot_without_label_excluded(self):
        """Addplots without ylabel= should not appear in the legend."""
        df = _sample_df()
        ap = wickly.make_addplot(df["Close"].rolling(5).mean(), color="red")
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        assert result is not None
        widget, _ = result
        assert widget._addplots[0].get("ylabel") is None
        _safe_close(widget)


class TestAppendData:
    """Tests for CandlestickWidget.append_data()."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def _make_widget(self, n: int = 30):
        df = _sample_df(n)
        result = wickly.plot(df, type="candle", volume=True, returnfig=True)
        assert result is not None
        return result[0], df

    def test_append_increases_bar_count(self):
        widget, df = self._make_widget()
        old_n = widget._n
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5], [5000.0])
        assert widget._n == old_n + 1
        _safe_close(widget)

    def test_append_multiple_bars(self):
        widget, df = self._make_widget()
        old_n = widget._n
        new_dates = pd.bdate_range("2025-03-01", periods=5)
        rng = np.random.default_rng(42)
        widget.append_data(
            new_dates,
            rng.normal(100, 1, 5),
            rng.normal(102, 1, 5),
            rng.normal(98, 1, 5),
            rng.normal(101, 1, 5),
            rng.integers(1000, 9999, 5).astype(float),
        )
        assert widget._n == old_n + 5
        _safe_close(widget)

    def test_append_auto_scroll(self):
        widget, df = self._make_widget()
        # Scroll to the end first
        widget._view_end = widget._n - 1
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5], [5000.0])
        assert widget._view_end == widget._n - 1
        _safe_close(widget)

    def test_append_no_auto_scroll(self):
        widget, df = self._make_widget()
        # Pan to the left so we're NOT at the right edge
        widget._view_start = 0
        widget._view_end = 10
        old_end = widget._view_end
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5], [5000.0],
                           auto_scroll=False)
        assert widget._view_end == old_end
        _safe_close(widget)

    def test_append_recomputes_mavs(self):
        df = _sample_df(30)
        result = wickly.plot(df, type="candle", mav=(5,), returnfig=True)
        assert result is not None
        widget, _ = result
        old_mav_len = len(widget._mav_data[0])
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5])
        assert len(widget._mav_data[0]) == old_mav_len + 1
        _safe_close(widget)

    def test_append_without_volume(self):
        df = _sample_df().drop(columns=["Volume"])
        result = wickly.plot(df, type="candle", returnfig=True)
        assert result is not None
        widget, _ = result
        old_n = widget._n
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5])
        assert widget._n == old_n + 1
        assert widget._volumes is None
        _safe_close(widget)


class TestUpdateLast:
    """Tests for CandlestickWidget.update_last()."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def _make_widget(self):
        df = _sample_df(30)
        result = wickly.plot(df, type="candle", volume=True, returnfig=True)
        assert result is not None
        return result[0], df

    def test_update_close(self):
        widget, df = self._make_widget()
        widget.update_last(close=999.0)
        assert widget._closes[-1] == 999.0
        _safe_close(widget)

    def test_update_high(self):
        widget, df = self._make_widget()
        widget.update_last(high=1234.0)
        assert widget._highs[-1] == 1234.0
        _safe_close(widget)

    def test_update_low(self):
        widget, df = self._make_widget()
        widget.update_last(low=0.5)
        assert widget._lows[-1] == 0.5
        _safe_close(widget)

    def test_update_open(self):
        widget, df = self._make_widget()
        widget.update_last(open_=50.0)
        assert widget._opens[-1] == 50.0
        _safe_close(widget)

    def test_update_volume(self):
        widget, df = self._make_widget()
        widget.update_last(volume=42.0)
        assert widget._volumes[-1] == 42.0
        _safe_close(widget)

    def test_update_preserves_other_fields(self):
        widget, df = self._make_widget()
        old_open = widget._opens[-1]
        old_high = widget._highs[-1]
        widget.update_last(close=999.0)
        assert widget._opens[-1] == old_open
        assert widget._highs[-1] == old_high
        _safe_close(widget)

    def test_update_noop_on_empty(self):
        """update_last on a widget with no data should not crash."""
        df = _sample_df(5)
        result = wickly.plot(df, type="candle", returnfig=True)
        assert result is not None
        widget, _ = result
        # Clear all data to simulate empty state
        widget._opens = np.array([])
        widget._highs = np.array([])
        widget._lows = np.array([])
        widget._closes = np.array([])
        widget._volumes = None
        widget._n = 0
        widget.update_last(close=100.0)  # should not raise
        assert widget._n == 0
        _safe_close(widget)

    def test_update_recomputes_mavs(self):
        df = _sample_df(30)
        result = wickly.plot(df, type="candle", mav=(5,), returnfig=True)
        assert result is not None
        widget, _ = result
        old_last_mav = widget._mav_data[0][-1]
        widget.update_last(close=999.0)
        new_last_mav = widget._mav_data[0][-1]
        assert new_last_mav != old_last_mav
        _safe_close(widget)


class TestLivePlot:
    """Tests for the top-level live_plot() function."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_returns_widget(self):
        df = _sample_df()
        widget, axes = wickly.live_plot(df, type="candle")
        assert isinstance(widget, CandlestickWidget)
        assert "main" in axes
        _safe_close(widget)

    def test_live_plot_with_kwargs(self):
        df = _sample_df()
        widget, axes = wickly.live_plot(
            df, type="candle", volume=True, mav=(5, 10), style="yahoo",
        )
        assert widget._n == len(df)
        assert len(widget._mavs) == 2
        _safe_close(widget)

    def test_live_plot_then_append(self):
        df = _sample_df(20)
        widget, _ = wickly.live_plot(df, type="candle", volume=True)
        old_n = widget._n
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-04-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5], [5000.0])
        assert widget._n == old_n + 1
        _safe_close(widget)

    def test_live_plot_then_update_last(self):
        df = _sample_df(20)
        widget, _ = wickly.live_plot(df, type="candle")
        widget.update_last(close=42.0)
        assert widget._closes[-1] == 42.0
        _safe_close(widget)


class TestLiveAddplot:
    """Tests for live addplot update methods."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def _make_widget_with_addplot(self, n=30):
        df = _sample_df(n)
        sma = df["Close"].rolling(5).mean()
        ap = wickly.make_addplot(sma, color="blue", ylabel="SMA 5")
        result = wickly.plot(df, type="candle", addplot=ap, returnfig=True)
        assert result is not None
        return result[0], df, sma

    # -- update_addplot --

    def test_update_addplot_replaces_data(self):
        widget, df, _ = self._make_widget_with_addplot()
        new_data = np.full(widget._n, 42.0)
        widget.update_addplot(0, new_data)
        np.testing.assert_array_equal(widget._addplots[0]["data"], new_data)
        _safe_close(widget)

    def test_update_addplot_with_series(self):
        widget, df, _ = self._make_widget_with_addplot()
        new_data = pd.Series(np.full(widget._n, 99.0))
        widget.update_addplot(0, new_data)
        assert widget._addplots[0]["data"][-1] == 99.0
        _safe_close(widget)

    def test_update_addplot_bad_index_raises(self):
        widget, df, _ = self._make_widget_with_addplot()
        with pytest.raises(IndexError, match="out of range"):
            widget.update_addplot(5, np.array([1.0]))
        _safe_close(widget)

    def test_update_addplot_segments(self):
        df = _sample_df(30)
        seg = wickly.make_segments([(5, [1.0, 2.0, 3.0])], color="red")
        result = wickly.plot(df, type="candle", addplot=seg, returnfig=True)
        assert result is not None
        widget, _ = result
        new_segs = [(10, [4.0, 5.0])]
        widget.update_addplot(0, new_segs)
        assert widget._addplots[0]["data"][0][0] == 10
        _safe_close(widget)

    # -- append_addplot_data --

    def test_append_addplot_data_single(self):
        widget, df, _ = self._make_widget_with_addplot()
        old_len = len(widget._addplots[0]["data"])
        widget.append_addplot_data(0, 123.0)
        assert len(widget._addplots[0]["data"]) == old_len + 1
        assert widget._addplots[0]["data"][-1] == 123.0
        _safe_close(widget)

    def test_append_addplot_data_array(self):
        widget, df, _ = self._make_widget_with_addplot()
        old_len = len(widget._addplots[0]["data"])
        widget.append_addplot_data(0, [1.0, 2.0, 3.0])
        assert len(widget._addplots[0]["data"]) == old_len + 3
        _safe_close(widget)

    def test_append_addplot_data_nan(self):
        widget, df, _ = self._make_widget_with_addplot()
        widget.append_addplot_data(0, float("nan"))
        assert np.isnan(widget._addplots[0]["data"][-1])
        _safe_close(widget)

    def test_append_addplot_data_segments_raises(self):
        df = _sample_df(30)
        seg = wickly.make_segments([(5, [1.0, 2.0])], color="red")
        result = wickly.plot(df, type="candle", addplot=seg, returnfig=True)
        assert result is not None
        widget, _ = result
        with pytest.raises(TypeError, match="does not support"):
            widget.append_addplot_data(0, 1.0)
        _safe_close(widget)

    def test_append_addplot_bad_index_raises(self):
        widget, df, _ = self._make_widget_with_addplot()
        with pytest.raises(IndexError, match="out of range"):
            widget.append_addplot_data(99, 1.0)
        _safe_close(widget)

    # -- update_addplot_last --

    def test_update_addplot_last(self):
        widget, df, _ = self._make_widget_with_addplot()
        widget.update_addplot_last(0, 777.0)
        assert widget._addplots[0]["data"][-1] == 777.0
        _safe_close(widget)

    def test_update_addplot_last_bad_index_raises(self):
        widget, df, _ = self._make_widget_with_addplot()
        with pytest.raises(IndexError, match="out of range"):
            widget.update_addplot_last(5, 1.0)
        _safe_close(widget)

    def test_update_addplot_last_segments_raises(self):
        df = _sample_df(30)
        seg = wickly.make_segments([(5, [1.0, 2.0])], color="red")
        result = wickly.plot(df, type="candle", addplot=seg, returnfig=True)
        assert result is not None
        widget, _ = result
        with pytest.raises(TypeError, match="does not support"):
            widget.update_addplot_last(0, 1.0)
        _safe_close(widget)

    # -- combined live flow --

    def test_live_append_with_addplot(self):
        """Append a bar AND an addplot point together, as in a live flow."""
        df = _sample_df(20)
        sma = df["Close"].rolling(5).mean()
        ap = wickly.make_addplot(sma, color="blue", ylabel="SMA 5")
        widget, _ = wickly.live_plot(df, type="candle", addplot=ap, volume=True)

        old_n = widget._n
        old_ap_len = len(widget._addplots[0]["data"])

        # Simulate new bar + new SMA point
        new_dates = pd.DatetimeIndex([pd.Timestamp("2025-04-01")])
        widget.append_data(new_dates, [100.0], [101.0], [99.0], [100.5], [5000.0])
        widget.append_addplot_data(0, 100.3)

        assert widget._n == old_n + 1
        assert len(widget._addplots[0]["data"]) == old_ap_len + 1
        assert widget._addplots[0]["data"][-1] == 100.3
        _safe_close(widget)

