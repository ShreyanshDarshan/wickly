# Wickly

**Interactive candlestick chart plotting with PyQt6** — drop-in replacement for `mplfinance` with rich interactivity.

## Features

- 📈 Candlestick, OHLC bar, line, and hollow-candle chart types
- 📊 Optional volume subplot
- 🔍 Mouse-wheel zoom and click-drag pan
- 🎯 Crosshair cursor with live OHLC / volume readout
- 📐 Moving average overlays (`mav` parameter)
- 🎨 Built-in styles (`default`, `charles`, `mike`, `yahoo`, `classic`, `nightclouds`)
- 🖼️ `make_addplot()` for overlaying custom data (line, scatter & segments)
- 🔀 `make_segments()` for overlapping broken-line overlays (e.g. Knoxville Divergence)
- 💾 `savefig()` support (PNG / JPG / BMP)
- 🔁 `returnfig` mode for embedding in your own PyQt6 application
- ✅ mplfinance-compatible function signatures

## Installation

```bash
pip install .
```

Or in editable/development mode:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
import wickly

# DataFrame must have a DatetimeIndex and columns: Open, High, Low, Close (and optionally Volume)
df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Candlestick chart with volume and moving averages — same API as mplfinance
wickly.plot(df, type='candle', volume=True, mav=(10, 20), style='yahoo', title='My Chart')
```

## API Reference

### `wickly.plot(data, **kwargs)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | *required* | OHLCV DataFrame with DatetimeIndex |
| `type` | str | `'ohlc'` | `'candle'`, `'ohlc'`, `'line'`, `'hollow'` |
| `style` | str or dict | `'default'` | Style name or dict of color settings |
| `volume` | bool | `False` | Show volume subplot |
| `mav` | int or tuple | `None` | Moving average period(s) |
| `title` | str | `None` | Chart title |
| `ylabel` | str | `'Price'` | Y-axis label |
| `figsize` | tuple | `(960, 600)` | Widget size in pixels `(width, height)` |
| `returnfig` | bool | `False` | Return `(widget, axes_dict)` instead of showing |
| `savefig` | str | `None` | Save chart to file path |
| `addplot` | dict/list | `None` | Additional plots (see `make_addplot`) |
| `columns` | tuple | `("Open","High","Low","Close","Volume")` | Column name mapping |
| `block` | bool | `True` | Block until window is closed |

### `wickly.make_addplot(data, **kwargs)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | Series/list | *required* | Data to overlay (or list of `(start, values)` tuples for `type='segments'`) |
| `type` | str | `'line'` | `'line'`, `'scatter'`, or `'segments'` |
| `color` | str | `None` | Line / marker color |
| `width` | float | `1.5` | Line width |
| `scatter` | bool | `False` | *Deprecated* — use `type='scatter'` |
| `marker` | str | `'o'` | Scatter marker character |
| `markersize` | float | `50` | Scatter marker size |
| `ylabel` | str | `None` | Legend label — if set, the overlay appears in the chart legend |

### `wickly.make_segments(segments, **kwargs)`

Convenience wrapper for `make_addplot(type='segments')` — builds an addplot dict
for multiple independent, possibly overlapping, line segments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segments` | list | *required* | List of `(start_index, y_values)` tuples |
| `color` | str | `None` | Line color |
| `width` | float | `1.5` | Line width |
| `alpha` | float | `1.0` | Opacity |
| `linestyle` | str | `'-'` | `'-'`, `'--'`, `'-.'`, `':'` |
| `ylabel` | str | `None` | Legend label |

### `wickly.make_style(**kwargs)`

Create a custom style dict with keys: `up_color`, `down_color`, `volume_up`, `volume_down`, `bg_color`, `grid_color`, `edge_up`, `edge_down`, `wick_up`, `wick_down`.

### `wickly.available_styles()`

Returns a list of available built-in style names.

## Contributing

Contributions are welcome — bug reports, feature requests, documentation improvements, and pull requests all appreciated.

### 1. Fork & clone

```bash
git clone https://github.com/your-org/wickly.git
cd wickly
```

### 2. Set up the development environment

Create and activate a dedicated environment (conda shown; plain venv works too):

```bash
conda create -n wickly python=3.11
conda activate wickly
pip install -e ".[dev,docs]"
```

### 3. Project layout

```
wickly/
├── src/wickly/          # Package source
│   ├── __init__.py      # Public API surface
│   ├── plotting.py      # Top-level plot() function
│   ├── chart_widget.py  # PyQt6 CandlestickWidget
│   ├── addplot.py       # make_addplot() & make_segments() helpers
│   ├── styles.py        # Built-in styles & make_style()
│   └── _utils.py        # Data validation utilities
├── tests/               # pytest test suite
├── docs/                # Sphinx documentation source
├── examples/            # Runnable example scripts
├── pyproject.toml       # Project metadata & dependencies
└── .readthedocs.yaml    # Read the Docs configuration
```

### 4. Run the tests

```bash
pytest tests/ -v
```

All 41 tests should pass. Add new tests under `tests/` for any code you contribute.

### 5. Build the documentation locally

```bash
# HTML output → docs/_build/html/index.html
python -m sphinx -b html docs docs/_build/html

# Open in the default browser (Windows)
start docs/_build/html/index.html

# Open in the default browser (macOS / Linux)
open docs/_build/html/index.html
```

Build must succeed with **zero warnings** (`-W` flag):

```bash
python -m sphinx -b html docs docs/_build/html -W
```

### 6. Run the examples

```bash
python examples/basic_candle.py        # Basic candlestick + volume + MAs
python examples/chart_types.py         # All four chart types in sequence
python examples/addplot_overlay.py     # Bollinger Bands & scatter signals
```

### 7. Adding a new built-in style

1. Add an entry to `_BUILTIN_STYLES` in `src/wickly/styles.py` following the
   existing key schema.
2. Add the style name to `TestAvailableStyles.test_all_builtins_present` in
   `tests/test_styles.py`.
3. Document it in the styles table in `docs/styles.rst`.

### 8. Code style

- Format with **black** (`pip install black; black src/ tests/`).
- Type-annotate all public functions.
- Write NumPy-style docstrings — they are picked up by Sphinx autodoc.

### 9. Submitting a pull request

1. Branch off `main`: `git checkout -b feat/my-feature`
2. Make your changes, add tests, update docs.
3. Ensure `pytest tests/ -v` is all green and `sphinx -W` builds cleanly.
4. Open a PR against `main` with a clear description of what changed and why.

## License

MIT
