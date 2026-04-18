"""Tests for the indicator registry and compute functions."""

import numpy as np
import pandas as pd
import pytest

import wickly
from wickly.indicators import (
    IndicatorSpec,
    OutputSpec,
    ParamSpec,
    categories,
    get_indicator,
    list_indicators,
    register_indicator,
)
from wickly.chart_widget import CandlestickWidget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.3, n),
            "High": close + abs(rng.normal(0, 0.5, n)) + 0.5,
            "Low": close - abs(rng.normal(0, 0.5, n)) - 0.5,
            "Close": close,
            "Volume": rng.integers(1000, 9999, n).astype(float),
        },
        index=pd.bdate_range("2025-01-01", periods=n),
    )


def _safe_close(widget) -> None:
    try:
        from PyQt6 import sip
        if not sip.isdeleted(widget):
            widget.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_builtins_registered(self):
        specs = list_indicators()
        names = {s.name for s in specs}
        assert "SMA" in names
        assert "RSI" in names
        assert "MACD" in names
        assert "BBANDS" in names

    def test_categories(self):
        cats = categories()
        assert "Trend" in cats
        assert "Momentum" in cats

    def test_get_indicator(self):
        spec = get_indicator("RSI")
        assert spec.name == "RSI"
        assert spec.overlay is False
        assert len(spec.params) == 1

    def test_get_indicator_missing(self):
        with pytest.raises(KeyError):
            get_indicator("NONEXISTENT")

    def test_list_by_category(self):
        trend = list_indicators("Trend")
        for s in trend:
            assert s.category == "Trend"

    def test_register_custom(self):
        def _dummy(closes, opens, highs, lows, volumes, **kw):
            return {"out": np.zeros_like(closes)}

        spec = IndicatorSpec(
            name="_TEST_CUSTOM",
            display_name="Test Custom",
            category="Test",
            overlay=True,
            compute=_dummy,
            outputs=[OutputSpec("out", "Test", "#000")],
        )
        register_indicator(spec)
        assert get_indicator("_TEST_CUSTOM").name == "_TEST_CUSTOM"


# ---------------------------------------------------------------------------
# Compute function tests
# ---------------------------------------------------------------------------

class TestCompute:
    @pytest.fixture()
    def ohlcv(self):
        df = _sample_df(100)
        return (
            df["Close"].values,
            df["Open"].values,
            df["High"].values,
            df["Low"].values,
            df["Volume"].values,
        )

    @pytest.mark.parametrize("name", [
        "SMA", "EMA", "BBANDS", "RSI", "MACD", "STOCH", "ROC",
        "WILLR", "CCI", "ATR", "OBV", "MFI", "VWAP",
    ])
    def test_builtin_output_keys(self, name, ohlcv):
        spec = get_indicator(name)
        result = spec.compute(*ohlcv)
        for out in spec.outputs:
            assert out.key in result, f"{name}: missing key '{out.key}'"
            assert len(result[out.key]) == len(ohlcv[0])

    def test_rsi_range(self, ohlcv):
        spec = get_indicator("RSI")
        result = spec.compute(*ohlcv)
        rsi = result["rsi"]
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_sma_matches_pandas(self, ohlcv):
        closes = ohlcv[0]
        spec = get_indicator("SMA")
        result = spec.compute(*ohlcv, period=10)
        expected = pd.Series(closes).rolling(10).mean().values
        np.testing.assert_allclose(result["sma"], expected, equal_nan=True)

    def test_macd_three_outputs(self, ohlcv):
        spec = get_indicator("MACD")
        result = spec.compute(*ohlcv)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        np.testing.assert_allclose(
            result["histogram"],
            result["macd"] - result["signal"],
        )

    def test_bbands_symmetry(self, ohlcv):
        spec = get_indicator("BBANDS")
        result = spec.compute(*ohlcv, period=20, std_dev=2.0)
        mid = result["middle"]
        upper_diff = result["upper"] - mid
        lower_diff = mid - result["lower"]
        np.testing.assert_allclose(upper_diff, lower_diff, equal_nan=True)


# ---------------------------------------------------------------------------
# Widget integration tests
# ---------------------------------------------------------------------------

class TestWidgetIntegration:
    def test_add_overlay_indicator(self, qapp):
        df = _sample_df(50)
        widget, axes = wickly.plot(df, type="candle", returnfig=True, block=False)
        try:
            n_before = len(widget._addplots)
            uid = widget.add_indicator("SMA", period=10)
            assert uid is not None
            assert len(widget._addplots) > n_before
            assert len(widget._active_indicators) == 1
        finally:
            _safe_close(widget)

    def test_add_panel_indicator(self, qapp):
        df = _sample_df(50)
        widget, axes = wickly.plot(df, type="candle", returnfig=True, block=False)
        try:
            n_panels_before = len(widget._sub_panels)
            uid = widget.add_indicator("RSI", period=14)
            assert len(widget._sub_panels) == n_panels_before + 1
            assert len(widget._active_indicators) == 1
        finally:
            _safe_close(widget)

    def test_remove_indicator(self, qapp):
        df = _sample_df(50)
        widget, axes = wickly.plot(df, type="candle", returnfig=True, block=False)
        try:
            uid = widget.add_indicator("RSI")
            n_panels = len(widget._sub_panels)
            widget.remove_indicator(uid)
            assert len(widget._sub_panels) == n_panels - 1
            assert len(widget._active_indicators) == 0
        finally:
            _safe_close(widget)

    def test_add_multiple(self, qapp):
        df = _sample_df(50)
        widget, axes = wickly.plot(df, type="candle", returnfig=True, block=False)
        try:
            uid1 = widget.add_indicator("SMA", period=10)
            uid2 = widget.add_indicator("SMA", period=20)
            uid3 = widget.add_indicator("RSI")
            assert len(widget._active_indicators) == 3
            active = widget.list_active_indicators()
            assert len(active) == 3
        finally:
            _safe_close(widget)

    def test_add_macd_panel(self, qapp):
        df = _sample_df(50)
        widget, axes = wickly.plot(df, type="candle", returnfig=True, block=False)
        try:
            uid = widget.add_indicator("MACD")
            assert len(widget._active_indicators) == 1
            ai = widget._active_indicators[0]
            assert ai.panel_index is not None
            panel = widget._sub_panels[ai.panel_index]
            assert panel.bar_color_mode == "macd"
            # MACD panel should have 2 sub-addplots (macd line + signal line)
            assert len(panel.addplots) >= 2
        finally:
            _safe_close(widget)

    def test_recompute_on_set_data(self, qapp):
        df = _sample_df(50)
        widget, axes = wickly.plot(df, type="candle", returnfig=True, block=False)
        try:
            widget.add_indicator("RSI")
            old_data = widget._sub_panels[-1].data.copy()
            # Replace data
            df2 = _sample_df(60)
            from wickly._utils import check_and_prepare_data
            o, h, l, c, v, d = check_and_prepare_data(df2)
            widget.set_data(d, o, h, l, c, v)
            new_data = widget._sub_panels[-1].data
            assert len(new_data) == 60
        finally:
            _safe_close(widget)
