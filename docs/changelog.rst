Changelog
=========

v0.1.1 (2026-04-14)
--------------------

*Documentation & distribution release. No API changes.*

- Docs now hosted on Read the Docs at https://wickly.readthedocs.io/
- Package now available on PyPI — ``pip install wickly``

v0.1.0 (2026-04-12)
--------------------

*Initial release.*

- Candlestick, OHLC-bar, line, and hollow-candle chart types
- Volume sub-chart
- Moving average overlays (``mav`` parameter)
- ``make_addplot()`` for line & scatter overlays
- Mouse-wheel zoom and click-drag pan
- Crosshair cursor with OHLC / volume tooltip
- Six built-in styles: ``default``, ``charles``, ``mike``, ``yahoo``,
  ``classic``, ``nightclouds``
- ``make_style()`` for custom palettes
- ``savefig`` support (PNG / JPG / BMP)
- ``returnfig`` mode for embedding in PyQt6 applications
- mplfinance-compatible function signatures
