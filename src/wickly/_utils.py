"""Data validation and preparation utilities."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd


def check_and_prepare_data(
    data: pd.DataFrame,
    columns: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, pd.DatetimeIndex]:
    """Validate an OHLC(V) DataFrame and return arrays.

    Returns
    -------
    opens, highs, lows, closes, volumes (or None), dates
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a Pandas DataFrame with a DatetimeIndex.")

    col_open, col_high, col_low, col_close, col_volume = columns

    # --- accept case-insensitive column lookup ---------------------------------
    col_map = {c.lower(): c for c in data.columns}
    def _resolve(name: str) -> str:
        if name in data.columns:
            return name
        low = name.lower()
        if low in col_map:
            return col_map[low]
        raise ValueError(
            f"Column '{name}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )

    col_open  = _resolve(col_open)
    col_high  = _resolve(col_high)
    col_low   = _resolve(col_low)
    col_close = _resolve(col_close)

    opens  = data[col_open].values.astype(float)
    highs  = data[col_high].values.astype(float)
    lows   = data[col_low].values.astype(float)
    closes = data[col_close].values.astype(float)

    volumes: np.ndarray | None = None
    try:
        col_vol = _resolve(col_volume)
        volumes = data[col_vol].values.astype(float)
    except ValueError:
        pass

    # Dates
    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    else:
        try:
            dates = pd.DatetimeIndex(data.index)
        except Exception:
            dates = pd.date_range("2000-01-01", periods=len(data), freq="B")

    return opens, highs, lows, closes, volumes, dates
