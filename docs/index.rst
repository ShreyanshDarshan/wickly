Wickly Documentation
====================

**Interactive candlestick chart plotting with PyQt6** — a drop-in replacement
for `mplfinance <https://github.com/matplotlib/mplfinance>`_ with rich
interactivity.

.. code-block:: python

   import wickly

   wickly.plot(df, type='candle', volume=True, mav=(10, 20), style='yahoo')

Key Features
------------

- 📈 **Chart types** — Candlestick, OHLC bar, line, and hollow-candle
- 📊 **Volume subplot** — Optional volume panel below the price chart
- 🔍 **Zoom & pan** — Mouse-wheel zoom and click-drag panning
- 🎯 **Crosshair** — Live OHLC / volume readout under the cursor
- 📐 **Moving averages** — Overlay any number of MAs with the ``mav`` parameter
- 🎨 **Styles** — Six built-in themes plus ``make_style()`` for custom palettes
- 🖼️ **Addplot** — Overlay custom line & scatter data
- 💾 **Export** — ``savefig`` support (PNG / JPG / BMP)
- 🔁 **Embedding** — ``returnfig`` mode for integration into your own PyQt6 app
- ✅ **mplfinance-compatible** — Familiar function signatures

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api
   styles
   examples
   changelog
