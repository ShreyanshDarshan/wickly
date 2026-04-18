"""Built-in indicator implementations (pure numpy / pandas)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wickly.indicators._registry import (
    IndicatorSpec,
    OutputSpec,
    ParamSpec,
    register_indicator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ewm(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential weighted mean matching ``pd.Series.ewm(span=...).mean()``."""
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).rolling(period).mean().values


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    prev_c = np.empty_like(closes)
    prev_c[0] = closes[0]
    prev_c[1:] = closes[:-1]
    return np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))


# ===================================================================
# Trend (overlay)
# ===================================================================

def _compute_sma(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 20,
) -> dict[str, np.ndarray]:
    return {"sma": _rolling_mean(closes, period)}


register_indicator(IndicatorSpec(
    name="SMA",
    display_name="Simple Moving Average",
    category="Trend",
    overlay=True,
    compute=_compute_sma,
    params=[ParamSpec("period", "Period", 20, 2, 500)],
    outputs=[OutputSpec("sma", "SMA", "#ff9800")],
))


def _compute_ema(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 20,
) -> dict[str, np.ndarray]:
    return {"ema": _ewm(closes, period)}


register_indicator(IndicatorSpec(
    name="EMA",
    display_name="Exponential Moving Average",
    category="Trend",
    overlay=True,
    compute=_compute_ema,
    params=[ParamSpec("period", "Period", 20, 2, 500)],
    outputs=[OutputSpec("ema", "EMA", "#2196f3")],
))


def _compute_bbands(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, np.ndarray]:
    s = pd.Series(closes)
    mid = s.rolling(period).mean().values
    std = s.rolling(period).std(ddof=0).values
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return {"upper": upper, "middle": mid, "lower": lower}


register_indicator(IndicatorSpec(
    name="BBANDS",
    display_name="Bollinger Bands",
    category="Trend",
    overlay=True,
    compute=_compute_bbands,
    params=[
        ParamSpec("period", "Period", 20, 2, 500),
        ParamSpec("std_dev", "Std Dev", 2.0, 0.5, 5.0, 0.5),
    ],
    outputs=[
        OutputSpec("upper", "BB Upper", "#9c27b0", linestyle="--"),
        OutputSpec("middle", "BB Middle", "#9c27b0"),
        OutputSpec("lower", "BB Lower", "#9c27b0", linestyle="--"),
    ],
))


# ===================================================================
# Momentum (sub-panel)
# ===================================================================

def _compute_rsi(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 14,
) -> dict[str, np.ndarray]:
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ewm(gain, period)
    avg_loss = _ewm(loss, period)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return {"rsi": rsi}


register_indicator(IndicatorSpec(
    name="RSI",
    display_name="Relative Strength Index",
    category="Momentum",
    overlay=False,
    compute=_compute_rsi,
    params=[ParamSpec("period", "Period", 14, 2, 100)],
    outputs=[OutputSpec("rsi", "RSI", "#ff9800")],
    ref_lines=[30.0, 70.0],
    height_ratio=0.15,
))


def _compute_macd(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, np.ndarray]:
    ema_fast = _ewm(closes, fast)
    ema_slow = _ewm(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ewm(macd_line, signal)
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


register_indicator(IndicatorSpec(
    name="MACD",
    display_name="MACD",
    category="Momentum",
    overlay=False,
    compute=_compute_macd,
    params=[
        ParamSpec("fast", "Fast Period", 12, 2, 100),
        ParamSpec("slow", "Slow Period", 26, 2, 200),
        ParamSpec("signal", "Signal Period", 9, 2, 50),
    ],
    outputs=[
        OutputSpec("macd", "MACD", "#2196f3"),
        OutputSpec("signal", "Signal", "#ff9800", linestyle="--"),
        OutputSpec("histogram", "Histogram", "#26a69a", plot_type="histogram"),
    ],
    zero_centered=True,
    height_ratio=0.18,
))


def _compute_stochastic(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, np.ndarray]:
    s_low = pd.Series(lows).rolling(k_period).min().values
    s_high = pd.Series(highs).rolling(k_period).max().values
    denom = s_high - s_low
    k = np.divide(closes - s_low, denom, out=np.full_like(closes, 50.0), where=denom != 0) * 100.0
    d = _rolling_mean(k, d_period)
    return {"k": k, "d": d}


register_indicator(IndicatorSpec(
    name="STOCH",
    display_name="Stochastic Oscillator",
    category="Momentum",
    overlay=False,
    compute=_compute_stochastic,
    params=[
        ParamSpec("k_period", "K Period", 14, 2, 100),
        ParamSpec("d_period", "D Period", 3, 2, 50),
    ],
    outputs=[
        OutputSpec("k", "%K", "#2196f3"),
        OutputSpec("d", "%D", "#ff9800", linestyle="--"),
    ],
    ref_lines=[20.0, 80.0],
    height_ratio=0.15,
))


def _compute_roc(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 12,
) -> dict[str, np.ndarray]:
    prev = np.empty_like(closes)
    prev[:period] = np.nan
    prev[period:] = closes[:-period]
    roc = np.divide(closes - prev, prev, out=np.full_like(closes, np.nan), where=prev != 0) * 100.0
    return {"roc": roc}


register_indicator(IndicatorSpec(
    name="ROC",
    display_name="Rate of Change",
    category="Momentum",
    overlay=False,
    compute=_compute_roc,
    params=[ParamSpec("period", "Period", 12, 2, 200)],
    outputs=[OutputSpec("roc", "ROC", "#e91e63")],
    zero_centered=True,
    height_ratio=0.15,
))


def _compute_willr(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 14,
) -> dict[str, np.ndarray]:
    hh = pd.Series(highs).rolling(period).max().values
    ll = pd.Series(lows).rolling(period).min().values
    denom = hh - ll
    wr = np.divide(hh - closes, denom, out=np.full_like(closes, -50.0), where=denom != 0) * -100.0
    return {"willr": wr}


register_indicator(IndicatorSpec(
    name="WILLR",
    display_name="Williams %R",
    category="Momentum",
    overlay=False,
    compute=_compute_willr,
    params=[ParamSpec("period", "Period", 14, 2, 100)],
    outputs=[OutputSpec("willr", "%R", "#9c27b0")],
    ref_lines=[-20.0, -80.0],
    height_ratio=0.15,
))


def _compute_cci(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 20,
) -> dict[str, np.ndarray]:
    tp = (highs + lows + closes) / 3.0
    s = pd.Series(tp)
    sma = s.rolling(period).mean().values
    mad = s.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).values
    cci = np.divide(tp - sma, 0.015 * mad, out=np.zeros_like(tp), where=mad != 0)
    return {"cci": cci}


register_indicator(IndicatorSpec(
    name="CCI",
    display_name="Commodity Channel Index",
    category="Momentum",
    overlay=False,
    compute=_compute_cci,
    params=[ParamSpec("period", "Period", 20, 2, 200)],
    outputs=[OutputSpec("cci", "CCI", "#00bcd4")],
    zero_centered=True,
    ref_lines=[-100.0, 100.0],
    height_ratio=0.15,
))


# ===================================================================
# Volatility (sub-panel)
# ===================================================================

def _compute_atr(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 14,
) -> dict[str, np.ndarray]:
    tr = _true_range(highs, lows, closes)
    atr = _ewm(tr, period)
    return {"atr": atr}


register_indicator(IndicatorSpec(
    name="ATR",
    display_name="Average True Range",
    category="Volatility",
    overlay=False,
    compute=_compute_atr,
    params=[ParamSpec("period", "Period", 14, 2, 100)],
    outputs=[OutputSpec("atr", "ATR", "#ff5722")],
    height_ratio=0.15,
))


# ===================================================================
# Volume (sub-panel)
# ===================================================================

def _compute_obv(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    **_: Any,
) -> dict[str, np.ndarray]:
    if volumes is None:
        return {"obv": np.zeros_like(closes)}
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    obv = np.cumsum(direction * volumes)
    return {"obv": obv}


register_indicator(IndicatorSpec(
    name="OBV",
    display_name="On-Balance Volume",
    category="Volume",
    overlay=False,
    compute=_compute_obv,
    params=[],
    outputs=[OutputSpec("obv", "OBV", "#607d8b")],
    height_ratio=0.15,
))


def _compute_mfi(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    *,
    period: int = 14,
) -> dict[str, np.ndarray]:
    if volumes is None:
        return {"mfi": np.full_like(closes, 50.0)}
    tp = (highs + lows + closes) / 3.0
    rmf = tp * volumes
    pos_flow = np.where(np.diff(tp, prepend=tp[0]) > 0, rmf, 0.0)
    neg_flow = np.where(np.diff(tp, prepend=tp[0]) < 0, rmf, 0.0)
    pos_sum = pd.Series(pos_flow).rolling(period).sum().values
    neg_sum = pd.Series(neg_flow).rolling(period).sum().values
    mr = np.divide(pos_sum, neg_sum, out=np.ones_like(pos_sum), where=neg_sum != 0)
    mfi = 100.0 - 100.0 / (1.0 + mr)
    return {"mfi": mfi}


register_indicator(IndicatorSpec(
    name="MFI",
    display_name="Money Flow Index",
    category="Volume",
    overlay=False,
    compute=_compute_mfi,
    params=[ParamSpec("period", "Period", 14, 2, 100)],
    outputs=[OutputSpec("mfi", "MFI", "#795548")],
    ref_lines=[20.0, 80.0],
    height_ratio=0.15,
))


def _compute_vwap(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray | None,
    **_: Any,
) -> dict[str, np.ndarray]:
    if volumes is None:
        return {"vwap": closes.copy()}
    tp = (highs + lows + closes) / 3.0
    cum_tpv = np.cumsum(tp * volumes)
    cum_vol = np.cumsum(volumes)
    vwap = np.divide(cum_tpv, cum_vol, out=np.zeros_like(cum_tpv), where=cum_vol != 0)
    return {"vwap": vwap}


register_indicator(IndicatorSpec(
    name="VWAP",
    display_name="Volume-Weighted Avg Price",
    category="Volume",
    overlay=True,
    compute=_compute_vwap,
    params=[],
    outputs=[OutputSpec("vwap", "VWAP", "#673ab7", linestyle="--")],
))
