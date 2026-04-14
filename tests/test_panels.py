"""Tests for SubPanel / make_panel / CandlestickWidget sub-panel API."""

import numpy as np
import pandas as pd
import pytest

import wickly
from wickly.addplot import SubPanel, make_panel
from wickly.chart_widget import CandlestickWidget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df(n: int = 30) -> pd.DataFrame:
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


def _widget(n=30, panels=None, show_volume=False) -> CandlestickWidget:
    df = _sample_df(n)
    w, _ = wickly.plot(df, returnfig=True, block=False,
                       volume=show_volume, panels=panels)
    return w


def _safe_close(widget) -> None:
    try:
        from PyQt6 import sip
        if not sip.isdeleted(widget):
            widget.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# make_panel validation
# ---------------------------------------------------------------------------

class TestMakePanel:
    def test_returns_subpanel(self):
        panel = make_panel([1.0, 2.0, 3.0])
        assert isinstance(panel, SubPanel)

    def test_list_input_converted(self):
        panel = make_panel([10.0, 20.0])
        assert isinstance(panel.data, np.ndarray)
        np.testing.assert_array_equal(panel.data, [10.0, 20.0])

    def test_series_input(self):
        s = pd.Series([1.0, 2.0])
        panel = make_panel(s)
        np.testing.assert_array_equal(panel.data, [1.0, 2.0])

    def test_numpy_input(self):
        arr = np.array([5.0, 6.0, 7.0])
        panel = make_panel(arr)
        np.testing.assert_array_equal(panel.data, arr)

    def test_defaults(self):
        panel = make_panel([1.0])
        assert panel.ylabel == "Value"
        assert panel.height_ratio == pytest.approx(0.20)
        assert panel.panel_type == "line"
        assert panel.visible is True

    def test_custom_ylabel(self):
        panel = make_panel([1.0], ylabel="RSI")
        assert panel.ylabel == "RSI"

    def test_custom_color(self):
        panel = make_panel([1.0], color="#ff0000")
        assert panel.color == "#ff0000"

    def test_default_color_is_set(self):
        panel = make_panel([1.0])
        assert panel.color is not None and len(panel.color) > 0

    def test_histogram_type(self):
        panel = make_panel([1.0], panel_type="histogram")
        assert panel.panel_type == "histogram"

    def test_invalid_panel_type_raises(self):
        with pytest.raises(ValueError, match="panel_type"):
            make_panel([1.0], panel_type="scatter")

    def test_height_ratio(self):
        panel = make_panel([1.0], height_ratio=0.30)
        assert panel.height_ratio == pytest.approx(0.30)

    def test_width(self):
        panel = make_panel([1.0], width=2.5)
        assert panel.width == pytest.approx(2.5)

    def test_alpha(self):
        panel = make_panel([1.0], alpha=0.5)
        assert panel.alpha == pytest.approx(0.5)

    def test_linestyle(self):
        panel = make_panel([1.0], linestyle="--")
        assert panel.linestyle == "--"

    def test_addplot_list(self):
        overlay = wickly.make_addplot([1.0, 2.0])
        panel = make_panel([1.0, 2.0], addplot=[overlay])
        assert len(panel.addplots) == 1

    def test_addplot_single_dict(self):
        overlay = wickly.make_addplot([1.0, 2.0])
        panel = make_panel([1.0, 2.0], addplot=overlay)
        assert len(panel.addplots) == 1


# ---------------------------------------------------------------------------
# SubPanel dataclass
# ---------------------------------------------------------------------------

class TestSubPanelDataclass:
    def test_direct_construction(self):
        data = np.array([1.0, 2.0])
        sp = SubPanel(data=data, ylabel="Test")
        assert sp.ylabel == "Test"
        np.testing.assert_array_equal(sp.data, data)

    def test_addplots_default_empty(self):
        sp = SubPanel(data=np.array([1.0]))
        assert sp.addplots == []

    def test_visible_default_true(self):
        sp = SubPanel(data=np.array([1.0]))
        assert sp.visible is True


# ---------------------------------------------------------------------------
# CandlestickWidget construction with panels
# ---------------------------------------------------------------------------

class TestWidgetWithPanels:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_construct_no_panels(self):
        w = _widget()
        assert len(w._sub_panels) == 0
        _safe_close(w)

    def test_construct_one_panel(self):
        rsi = np.linspace(30, 70, 30)
        panel = make_panel(rsi, ylabel="RSI")
        w = _widget(panels=[panel])
        assert len(w._sub_panels) == 1
        _safe_close(w)

    def test_construct_two_panels(self):
        data = np.linspace(0, 1, 30)
        p1 = make_panel(data, ylabel="P1")
        p2 = make_panel(data, ylabel="P2", panel_type="histogram")
        w = _widget(panels=[p1, p2])
        assert len(w._sub_panels) == 2
        _safe_close(w)

    def test_panels_kwarg_via_plot(self):
        rsi = np.linspace(30, 70, 30)
        panel = make_panel(rsi, ylabel="RSI")
        w, _ = wickly.plot(_sample_df(), returnfig=True, block=False, panels=[panel])
        assert len(w._sub_panels) == 1
        _safe_close(w)

    def test_panels_kwarg_single_panel(self):
        rsi = np.linspace(30, 70, 30)
        panel = make_panel(rsi, ylabel="RSI")
        w, _ = wickly.plot(_sample_df(), returnfig=True, block=False, panels=panel)
        assert len(w._sub_panels) == 1
        _safe_close(w)


# ---------------------------------------------------------------------------
# Public panel API
# ---------------------------------------------------------------------------

class TestPanelAPI:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def _make_widget(self) -> CandlestickWidget:
        return _widget()

    def test_add_panel_returns_index(self):
        w = self._make_widget()
        panel = make_panel(np.zeros(30))
        idx = w.add_panel(panel)
        assert idx == 0
        assert len(w._sub_panels) == 1
        _safe_close(w)

    def test_add_multiple_panels(self):
        w = self._make_widget()
        w.add_panel(make_panel(np.zeros(30), ylabel="A"))
        idx = w.add_panel(make_panel(np.ones(30), ylabel="B"))
        assert idx == 1
        assert len(w._sub_panels) == 2
        _safe_close(w)

    def test_remove_panel(self):
        w = self._make_widget()
        w.add_panel(make_panel(np.zeros(30), ylabel="A"))
        w.add_panel(make_panel(np.ones(30), ylabel="B"))
        w.remove_panel(0)
        assert len(w._sub_panels) == 1
        assert w._sub_panels[0].ylabel == "B"
        _safe_close(w)

    def test_set_panel_data(self):
        w = self._make_widget()
        w.add_panel(make_panel(np.zeros(30)))
        new_data = np.ones(30) * 50
        w.set_panel_data(0, new_data)
        np.testing.assert_array_equal(w._sub_panels[0].data, new_data)
        _safe_close(w)

    def test_append_panel_data(self):
        w = self._make_widget()
        w.add_panel(make_panel(np.array([1.0, 2.0, 3.0])))
        w.append_panel_data(0, 4.0)
        assert len(w._sub_panels[0].data) == 4
        assert w._sub_panels[0].data[-1] == pytest.approx(4.0)
        _safe_close(w)

    def test_update_panel_last(self):
        w = self._make_widget()
        w.add_panel(make_panel(np.array([1.0, 2.0, 3.0])))
        w.update_panel_last(0, 99.0)
        assert w._sub_panels[0].data[-1] == pytest.approx(99.0)
        assert len(w._sub_panels[0].data) == 3
        _safe_close(w)

    def test_set_series_visible_panel(self):
        w = self._make_widget()
        w.add_panel(make_panel(np.zeros(30)))
        w.set_series_visible("panel", 0, False)
        assert w._sub_panels[0].visible is False
        w.set_series_visible("panel", 0, True)
        assert w._sub_panels[0].visible is True
        _safe_close(w)

    def test_set_series_visible_invalid_kind(self):
        w = self._make_widget()
        with pytest.raises(ValueError, match="Unknown kind"):
            w.set_series_visible("unknown_kind", 0, True)
        _safe_close(w)


# ---------------------------------------------------------------------------
# _layout_panels geometry
# ---------------------------------------------------------------------------

class TestLayoutPanels:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.closeAllWindows()

    def test_no_panels_main_fills_usable(self):
        w = _widget()
        w.resize(800, 600)
        mr, vr, prs = w._layout_panels()
        assert vr is None
        assert prs == []
        # Main rect height should equal usable height
        from wickly.chart_widget import CandlestickWidget
        usable = 600 - CandlestickWidget.MARGIN_TOP - CandlestickWidget.MARGIN_BOTTOM
        assert mr.height() == pytest.approx(usable, abs=1)
        _safe_close(w)

    def test_volume_shrinks_main(self):
        w = _widget(show_volume=True)
        w.resize(800, 600)
        mr, vr, prs = w._layout_panels()
        assert vr is not None
        assert mr.height() < 600
        _safe_close(w)

    def test_panel_appears_below_main(self):
        data = np.linspace(30, 70, 30)
        w = _widget(panels=[make_panel(data)])
        w.resize(800, 600)
        mr, vr, prs = w._layout_panels()
        assert len(prs) == 1
        assert prs[0].y() >= mr.y() + mr.height() - 1
        _safe_close(w)

    def test_hidden_panel_has_zero_height(self):
        data = np.linspace(30, 70, 30)
        p = make_panel(data)
        p.visible = False
        w = _widget(panels=[p])
        w.resize(800, 600)
        mr, vr, prs = w._layout_panels()
        assert prs[0].height() == pytest.approx(0.0)
        _safe_close(w)

    def test_height_ratio_clamping(self):
        """Even absurdly large ratios shouldn't let sub-panels steal >75 %."""
        panels = [make_panel(np.zeros(30), height_ratio=0.50) for _ in range(5)]
        w = _widget(panels=panels)
        w.resize(800, 600)
        mr, vr, prs = w._layout_panels()
        from wickly.chart_widget import CandlestickWidget
        usable = 600 - CandlestickWidget.MARGIN_TOP - CandlestickWidget.MARGIN_BOTTOM
        assert mr.height() >= usable * 0.25 - 1
        _safe_close(w)

    def test_two_panels_stack_vertically(self):
        p1 = make_panel(np.zeros(30), height_ratio=0.20)
        p2 = make_panel(np.ones(30), height_ratio=0.20)
        w = _widget(panels=[p1, p2])
        w.resize(800, 600)
        mr, vr, prs = w._layout_panels()
        assert len(prs) == 2
        # p2 starts where p1 ends
        assert prs[1].y() == pytest.approx(prs[0].y() + prs[0].height(), abs=1)
        _safe_close(w)
