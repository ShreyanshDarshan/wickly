![wickly logo](https://github.com/ShreyanshDarshan/wickly/raw/main/docs/_static/wickly_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/ShreyanshDarshan/wickly/actions/workflows/tests.yml/badge.svg)](https://github.com/ShreyanshDarshan/wickly/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/wickly/badge/?version=latest)](https://wickly.readthedocs.io/)


**Interactive candlestick chart plotting with PyQt6** — drop-in replacement for `mplfinance` with rich interactivity.

![wickly demo](https://github.com/ShreyanshDarshan/wickly/raw/main/docs/_static/wickly_demo.gif)

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
- 🔴 `live_plot()` for animated / live-updating charts
- ✅ mplfinance-compatible function signatures

## Installation

### Install from PyPI

The recommended way to install wickly:

```bash
pip install wickly
```

### Install from source

Clone the repository and install with pip:

```bash
git clone https://github.com/ShreyanshDarshan/wickly.git
cd wickly
pip install .
```

Or in editable/development mode (includes pytest and build):

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

## Documentation

Full API reference, quickstart guide, and examples are available at **[wickly.readthedocs.io](https://wickly.readthedocs.io/)**.

## Contributing

Contributions are welcome — bug reports, feature requests, documentation improvements, and pull requests all appreciated.

### 1. Fork & clone

```bash
git clone https://github.com/ShreyanshDarshan/wickly.git
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
python examples/live_chart.py          # Live / animated chart demo
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
