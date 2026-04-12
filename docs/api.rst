API Reference
=============

This page documents every public function and class in Wickly.

----

Top-level functions
-------------------

.. module:: wickly

.. autofunction:: wickly.plot

.. autofunction:: wickly.make_addplot

.. autofunction:: wickly.make_segments

.. autofunction:: wickly.make_style

.. autofunction:: wickly.available_styles

----

Chart widget
------------

.. module:: wickly.chart_widget

.. autoclass:: wickly.chart_widget.CandlestickWidget
   :members: set_data, reset_view, save
   :no-undoc-members:
   :show-inheritance:
   :exclude-members: rangeChanged

----

Plotting module
---------------

.. module:: wickly.plotting

.. autofunction:: wickly.plotting.plot

----

Addplot module
--------------

.. module:: wickly.addplot

.. autofunction:: wickly.addplot.make_addplot

.. autofunction:: wickly.addplot.make_segments

----

Styles module
-------------

.. module:: wickly.styles

.. autofunction:: wickly.styles.available_styles

.. autofunction:: wickly.styles.make_style

----

Utilities
---------

.. module:: wickly._utils

.. autofunction:: wickly._utils.check_and_prepare_data
