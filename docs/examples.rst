Examples
========

Each example can be run directly from the repository root.

Basic candlestick chart
-----------------------

A minimal chart with volume and two moving averages:

.. code-block:: bash

   python examples/basic_candle.py

.. literalinclude:: ../examples/basic_candle.py
   :language: python
   :caption: examples/basic_candle.py

Chart types
-----------

Cycle through all four chart types — ``candle``, ``ohlc``, ``line``,
``hollow``:

.. code-block:: bash

   python examples/chart_types.py

.. literalinclude:: ../examples/chart_types.py
   :language: python
   :caption: examples/chart_types.py

Addplot overlays (Bollinger Bands + Knoxville Divergence)
----------------------------------------------------------

Overlay Bollinger Bands, scatter-plot buy signals, and Knoxville Divergence
broken-line segments on top of a candlestick chart:

.. code-block:: bash

   python examples/addplot_overlay.py

.. literalinclude:: ../examples/addplot_overlay.py
   :language: python
   :caption: examples/addplot_overlay.py

