Installation
============

Requirements
------------

- Python ≥ 3.9
- PyQt6 ≥ 6.4
- pandas ≥ 1.5
- numpy ≥ 1.23

Install from PyPI
-----------------

The recommended way to install wickly:

.. code-block:: bash

   pip install wickly

Install from source
-------------------

Clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/ShreyanshDarshan/wickly.git
   cd wickly
   pip install .

Editable / development mode
----------------------------

For development work (includes ``pytest`` and ``build``):

.. code-block:: bash

   pip install -e ".[dev]"

To also install the documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Using Conda
-----------

If you use Conda, create an environment first:

.. code-block:: bash

   conda create -n wickly python=3.11
   conda activate wickly
   pip install -e ".[dev,docs]"

Verify the installation
-----------------------

.. code-block:: python

   import wickly
   print(wickly.__version__)
   print(wickly.available_styles())
