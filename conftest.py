"""Pytest config: put vendored vbjax/apvbt on sys.path before collection.

This mirrors the sys.path prepend in ``tvb_max/__init__.py`` so that
``import vbjax`` / ``import apvbt`` resolve to the vendored copies even
when a test imports them before importing tvb_max.
"""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
