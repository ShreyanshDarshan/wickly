# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the src directory so autodoc can import wickly
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Project information -----------------------------------------------------
project = "Wickly"
copyright = "2026, Wickly Contributors"
author = "Wickly Contributors"

# The version info for the project
from wickly._version import __version__

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Source file suffixes ----------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Mock PyQt6 only when it's not installed (e.g. on Read the Docs)
try:
    import PyQt6  # noqa: F401
except ImportError:
    autodoc_mock_imports = ["PyQt6"]

# Suppress RST formatting warnings from auto-generated pyqtSignal docstrings
suppress_warnings = ["autodoc.import"]
nitpicky = False

# -- Napoleon settings (Google / NumPy docstring support) --------------------
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

html_title = "Wickly"
html_short_title = "Wickly"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#26a69a",
        "color-brand-content": "#26a69a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#00e676",
        "color-brand-content": "#00e676",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_static_path = ["_static"]

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True
