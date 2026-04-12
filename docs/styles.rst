Styles
======

Wickly ships with **six built-in styles** and a ``make_style()`` helper for
creating custom palettes — mirroring the mplfinance styling API.

Built-in styles
---------------

Use any of the following names with the ``style`` parameter of
:func:`wickly.plot`:

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 28

   * - Style
     - Up colour
     - Down colour
     - Background
     - Notes
   * - ``"default"``
     - |def_up| ``#26a69a``
     - |def_dn| ``#ef5350``
     - ``#ffffff``
     - Teal / red on white
   * - ``"charles"``
     - |cha_up| ``#006340``
     - |cha_dn| ``#a02128``
     - ``#ffffff``
     - Classic green / maroon
   * - ``"mike"``
     - ``#000000``
     - |mik_dn| ``#0080ff``
     - |mik_bg| ``#0a0a23``
     - Dark-mode blue theme
   * - ``"yahoo"``
     - |yah_up| ``#00c853``
     - |yah_dn| ``#ff1744``
     - ``#ffffff``
     - Yahoo Finance colours
   * - ``"classic"``
     - ``#ffffff``
     - ``#000000``
     - ``#ffffff``
     - Black-and-white
   * - ``"nightclouds"``
     - |nig_up| ``#00e676``
     - |nig_dn| ``#ff5252``
     - |nig_bg| ``#1e1e2f``
     - Dark-mode green / red

.. |def_up| unicode:: U+1F7E2
.. |def_dn| unicode:: U+1F534
.. |cha_up| unicode:: U+1F7E2
.. |cha_dn| unicode:: U+1F534
.. |mik_dn| unicode:: U+1F535
.. |mik_bg| unicode:: U+26AB
.. |yah_up| unicode:: U+1F7E2
.. |yah_dn| unicode:: U+1F534
.. |nig_up| unicode:: U+1F7E2
.. |nig_dn| unicode:: U+1F534
.. |nig_bg| unicode:: U+26AB

Listing all available styles
----------------------------

.. code-block:: python

   import wickly
   print(wickly.available_styles())
   # ['default', 'charles', 'mike', 'yahoo', 'classic', 'nightclouds']

Using a style
-------------

Pass the style name as a string:

.. code-block:: python

   wickly.plot(df, type="candle", style="nightclouds")

Creating a custom style
------------------------

Use :func:`wickly.make_style` to derive a new style from any built-in
base:

.. code-block:: python

   my_style = wickly.make_style(
       base_mpf_style="nightclouds",
       up_color="#00ff00",
       down_color="#ff0000",
       bg_color="#111111",
       grid_color="#222222",
       mavcolors=["cyan", "magenta", "yellow"],
   )

   wickly.plot(df, style=my_style)

Style dictionary keys
---------------------

Every style is a plain ``dict`` with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``up_color``
     - Candle body fill when close ≥ open
   * - ``down_color``
     - Candle body fill when close < open
   * - ``edge_up``
     - Candle border colour (up)
   * - ``edge_down``
     - Candle border colour (down)
   * - ``wick_up``
     - Wick colour (up)
   * - ``wick_down``
     - Wick colour (down)
   * - ``volume_up``
     - Volume bar colour (up)
   * - ``volume_down``
     - Volume bar colour (down)
   * - ``bg_color``
     - Chart background colour
   * - ``grid_color``
     - Grid line colour
   * - ``text_color``
     - Axis label / title colour
   * - ``alpha``
     - Overall opacity (0.0 – 1.0)
   * - ``mavcolors``
     - List of colours for moving-average lines

You can also pass a raw ``dict`` directly to the ``style`` parameter of
``plot()`` — missing keys are filled from the ``"default"`` style.
