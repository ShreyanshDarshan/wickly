"""Shared pytest fixtures for the wickly test suite."""

import sys
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Session-scoped QApplication that stays alive for all tests.

    Without this, dropping the ``{"app": app}`` dict returned by ``plot()``
    can allow the *C++* QApplication to be garbage-collected, which destroys
    all top-level QWidgets and causes ``RuntimeError: wrapped C/C++ object
    of type CandlestickWidget has been deleted`` when any Qt method is called
    afterwards.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    yield app
