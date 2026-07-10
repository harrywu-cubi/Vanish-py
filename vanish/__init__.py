# vanish/__init__.py
"""Vanish-py: content-aware image resizing via seam carving."""
from vanish.features import resize, enlarge, remove_object
from vanish.carver import Carver

__all__ = ["resize", "enlarge", "remove_object", "Carver"]
