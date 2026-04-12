Quick Start
===========

This page walks through the most common usage patterns. Wickly's API mirrors
`mplfinance <https://github.com/matplotlib/mplfinance>`_ so if you're already
familiar with that library you can jump right in.

Prepare your data
-----------------

Wickly expects a ``pandas.DataFrame`` with a ``DatetimeIndex`` and columns
named **Open**, **High**, **Low**, **Close** (case-insensitive), plus an
optional **Volume** column.

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("ohlcv_data.csv", index_col=0, parse_dates=True)
   print(df.head())

Basic candlestick chart
-----------------------

.. code-block:: python

   import wickly

   wickly.plot(df, type="candle")

This opens an interactive window. Use the **mouse wheel** to zoom and
**click-drag** to pan.

Add volume & moving averages
-----------------------------

.. code-block:: python

   wickly.plot(
       df,
       type="candle",
       volume=True,
       mav=(10, 20),
       style="yahoo",
       title="My Chart",
   )

Use ``returnfig`` for embedding
-------------------------------

When building a larger PyQt6 application, use ``returnfig=True`` to get back
the widget without blocking:

.. code-block:: python

   widget, axes = wickly.plot(df, type="candle", returnfig=True)
   # widget is a QWidget you can embed in your own layout

Overlay custom data with ``make_addplot``
-----------------------------------------

.. code-block:: python

   import numpy as np

   sma = df["Close"].rolling(20).mean()
   upper = sma + 2 * df["Close"].rolling(20).std()
   lower = sma - 2 * df["Close"].rolling(20).std()

   signal = np.where(df["Close"] < lower, df["Low"] - 0.5, np.nan)

   apds = [
       wickly.make_addplot(upper, color="#9c27b0", linestyle="--"),
       wickly.make_addplot(lower, color="#9c27b0", linestyle="--"),
       wickly.make_addplot(signal, type="scatter", color="orange", markersize=90),
   ]

   wickly.plot(df, type="candle", volume=True, addplot=apds, style="nightclouds")

Save chart to file
------------------

.. code-block:: python

   wickly.plot(df, type="candle", savefig="chart.png")

Custom styles
-------------

.. code-block:: python

   my_style = wickly.make_style(
       base_mpf_style="nightclouds",
       up_color="#00ff00",
       down_color="#ff0000",
   )
   wickly.plot(df, style=my_style)

See :doc:`styles` for the full list of built-in themes and customisation
options.
