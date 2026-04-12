"""Tests for wickly.addplot module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from wickly.addplot import make_addplot


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
