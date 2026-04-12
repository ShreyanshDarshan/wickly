"""Tests for wickly.addplot module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from wickly.addplot import make_addplot, make_segments


class TestMakeAddplotBasic:
    def test_returns_dict(self):
        data = [1.0, 2.0, 3.0]
        result = make_addplot(data)
        assert isinstance(result, dict)

    def test_default_type_is_line(self):
        result = make_addplot([1, 2, 3])
        assert result["type"] == "line"

    def test_scatter_type(self):
        result = make_addplot([1, 2], type="scatter")
        assert result["type"] == "scatter"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="addplot type must be one of"):
            make_addplot([1], type="bar")

    def test_scatter_deprecated_kwarg(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = make_addplot([1, 2], scatter=True)
            assert result["type"] == "scatter"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


class TestMakeAddplotData:
    def test_list_input(self):
        result = make_addplot([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result["data"], [1.0, 2.0, 3.0])

    def test_numpy_input(self):
        arr = np.array([4.0, 5.0])
        result = make_addplot(arr)
        np.testing.assert_array_equal(result["data"], arr)

    def test_series_input(self):
        s = pd.Series([10.0, 20.0, 30.0])
        result = make_addplot(s)
        np.testing.assert_array_equal(result["data"], [10.0, 20.0, 30.0])

    def test_single_col_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = make_addplot(df)
        np.testing.assert_array_equal(result["data"], [1.0, 2.0])

    def test_multi_col_dataframe_raises(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="more than one column"):
            make_addplot(df)


class TestMakeAddplotParams:
    def test_color(self):
        result = make_addplot([1], color="#ff0000")
        assert result["color"] == "#ff0000"

    def test_width(self):
        result = make_addplot([1], width=3.0)
        assert result["width"] == 3.0

    def test_marker(self):
        result = make_addplot([1], type="scatter", marker="^")
        assert result["marker"] == "^"

    def test_markersize(self):
        result = make_addplot([1], type="scatter", markersize=200)
        assert result["markersize"] == 200

    def test_label_default_none(self):
        result = make_addplot([1, 2])
        assert result["ylabel"] is None

    def test_label_set(self):
        result = make_addplot([1, 2], ylabel="Upper BB")
        assert result["ylabel"] == "Upper BB"


class TestSegmentsType:
    """Tests for type='segments' in make_addplot."""

    def test_segments_returns_dict(self):
        segs = [(0, [1.0, 2.0, 3.0]), (5, [4.0, 5.0])]
        result = make_addplot(segs, type="segments")
        assert isinstance(result, dict)
        assert result["type"] == "segments"

    def test_segments_data_is_list_of_tuples(self):
        segs = [(0, [1.0, 2.0]), (3, [4.0])]
        result = make_addplot(segs, type="segments")
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 2
        start, vals = result["data"][0]
        assert start == 0
        np.testing.assert_array_equal(vals, [1.0, 2.0])

    def test_segments_normalises_to_float(self):
        segs = [(0, pd.Series([10, 20]))]
        result = make_addplot(segs, type="segments")
        _, vals = result["data"][0]
        assert vals.dtype == np.float64

    def test_segments_requires_list(self):
        with pytest.raises(TypeError, match="list of"):
            make_addplot(np.array([1.0, 2.0]), type="segments")

    def test_segments_bad_tuple_raises(self):
        with pytest.raises(ValueError, match="start_index, values"):
            make_addplot([(0, [1.0], "extra")], type="segments")

    def test_segments_overlapping_allowed(self):
        segs = [(5, [1.0, 2.0, 3.0]), (6, [10.0, 20.0, 30.0])]
        result = make_addplot(segs, type="segments")
        assert len(result["data"]) == 2

    def test_segments_preserves_style_kwargs(self):
        segs = [(0, [1.0])]
        result = make_addplot(segs, type="segments", color="#ff0000", width=3.0,
                              linestyle="--", alpha=0.6)
        assert result["color"] == "#ff0000"
        assert result["width"] == 3.0
        assert result["linestyle"] == "--"
        assert result["alpha"] == 0.6

    def test_segments_label(self):
        segs = [(0, [1.0, 2.0])]
        result = make_addplot(segs, type="segments", ylabel="Divergence")
        assert result["ylabel"] == "Divergence"


class TestMakeSegments:
    """Tests for the make_segments convenience function."""

    def test_returns_segments_dict(self):
        segs = [(0, [1.0, 2.0]), (5, [3.0, 4.0])]
        result = make_segments(segs, color="#e91e63")
        assert result["type"] == "segments"
        assert result["color"] == "#e91e63"

    def test_default_width(self):
        result = make_segments([(0, [1.0])])
        assert result["width"] == 1.5

    def test_linestyle_forwarded(self):
        result = make_segments([(0, [1.0])], linestyle="--")
        assert result["linestyle"] == "--"

    def test_data_contents(self):
        seg = (10, [50.1, 50.5, 51.0])
        result = make_segments([seg])
        start, vals = result["data"][0]
        assert start == 10
        np.testing.assert_array_almost_equal(vals, [50.1, 50.5, 51.0])

    def test_label_forwarded(self):
        result = make_segments([(0, [1.0])], ylabel="Knox Div")
        assert result["ylabel"] == "Knox Div"

    def test_label_default_none(self):
        result = make_segments([(0, [1.0])])
        assert result["ylabel"] is None
