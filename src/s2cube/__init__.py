# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""s2cube: build co-registered, leakage-safe Sentinel-2 time-series datasets for machine learning.

Modules
-------
config    project configuration (YAML)
geo       grids, CRS helpers, window sampling
stac      catalogue search and scene provenance
reader    reading one date onto a grid (harmonised, no interpolation of 20 m bands)
cube      building and reading the HDF5 cube
coreg     measuring and correcting per-date geolocation offsets
labels    pixel labels with ignore pixels and per-period label-drift checks
splits    spatially separated folds and nested label budgets
qa        reports and integrity manifest
datasets  NumPy datasets usable with PyTorch DataLoader
"""
from ._version import __version__
from .config import Config
from .cube import Cube, build, reflectance

__all__ = ["__version__", "Config", "Cube", "build", "reflectance"]
