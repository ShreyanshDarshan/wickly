"""Tests for wickly._utils module."""

import numpy as np
import pandas as pd
import pytest

from wickly._utils import check_and_prepare_data


def _sample_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


class TestCheckAndPrepareData:
    def test_basic(self):
        df = _sample_df()
        o, h, l, c, v, dates = check_and_prepare_data(df)
        assert len(o) == len(df)
        assert v is not None

    def test_case_insensitive_columns(self):
        df = _sample_df()
        df.columns = [c.lower() for c in df.columns]
        o, h, l, c, v, dates = check_and_prepare_data(df)
        assert len(o) == len(df)

    def test_missing_volume(self):
        df = _sample_df().drop(columns=["Volume"])
        o, h, l, c, v, dates = check_and_prepare_data(df)
        assert v is None

    def test_missing_required_column_raises(self):
        df = _sample_df().drop(columns=["High"])
        with pytest.raises(ValueError, match="Column 'High' not found"):
            check_and_prepare_data(df)

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            check_and_prepare_data([1, 2, 3])  # type: ignore[arg-type]

    def test_custom_columns(self):
        df = _sample_df()
        df.columns = ["o", "h", "l", "c", "v"]
        o, h, l, c, v, dates = check_and_prepare_data(df, columns=("o", "h", "l", "c", "v"))
        assert len(o) == len(df)

    def test_dates_returned(self):
        df = _sample_df()
        *_, dates = check_and_prepare_data(df)
        assert isinstance(dates, pd.DatetimeIndex)
        assert len(dates) == len(df)
