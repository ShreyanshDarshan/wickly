"""Tests for the top-level wickly.plot() function (non-GUI smoke tests)."""

import numpy as np
import pandas as pd
import pytest

import wickly
from wickly.plotting import _VALID_TYPES


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

