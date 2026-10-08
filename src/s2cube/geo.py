# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Coordinate reference systems, pixel grids and sampling of unlabelled windows.

All geometry inside s2cube is expressed in one projected CRS (by default the UTM zone of
the area of interest), in metres. A :class:`Grid` is an axis-aligned block of square pixels
whose corners are snapped to a multiple of the coarsest band resolution, so that 20 m
Sentinel-2 bands can be replicated 2x2 onto the 10 m grid without interpolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as _transform
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

WGS84 = CRS.from_epsg(4326)


def utm_crs(lon: float, lat: float) -> CRS:
    """UTM zone (WGS 84) containing a longitude/latitude point."""
    zone = min(int((lon + 180) // 6) + 1, 60)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def to_crs(geom, src, dst):
    """Reproject a shapely geometry from ``src`` to ``dst`` (anything accepted by CRS)."""
    src, dst = CRS.from_user_input(src), CRS.from_user_input(dst)
    if src == dst:
        return geom
    return shape(transform_geom(src, dst, mapping(geom)))


def points_to_lonlat(xs, ys, crs) -> tuple[np.ndarray, np.ndarray]:
    """Projected coordinates (arrays of any shape) to longitude/latitude arrays."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    lon, lat = _transform(CRS.from_user_input(crs), WGS84, xs.ravel().tolist(), ys.ravel().tolist())
    return np.reshape(lon, xs.shape), np.reshape(lat, ys.shape)


def snap_bounds(bounds, step: float) -> tuple[float, float, float, float]:
    """Expand (xmin, ymin, xmax, ymax) outwards to multiples of ``step``."""
    x0, y0, x1, y1 = bounds
    return (math.floor(x0 / step) * step, math.floor(y0 / step) * step,
            math.ceil(x1 / step) * step, math.ceil(y1 / step) * step)


@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid: upper-left corner (x0, y1), size in pixels and pixel size."""

    x0: float
    y1: float
    width: int
    height: int
    res: float

    @classmethod
    def from_bounds(cls, bounds, res: float, snap: float | None = None) -> Grid:
        """Grid covering ``bounds``; corners snapped outwards to ``snap`` (default ``res``)."""
        x0, y0, x1, y1 = snap_bounds(bounds, snap or res)
        return cls(x0, y1, int(round((x1 - x0) / res)), int(round((y1 - y0) / res)), res)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y1 - self.height * self.res, self.x0 + self.width * self.res, self.y1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y1)

    @property
    def geometry(self):
        return box(*self.bounds)

    def with_res(self, res: float) -> Grid:
        """Same extent at another pixel size (extent must be a multiple of ``res``)."""
        w, h = self.width * self.res / res, self.height * self.res / res
        if abs(w - round(w)) > 1e-6 or abs(h - round(h)) > 1e-6:
            raise ValueError(f"grid extent is not a multiple of {res} m")
        return Grid(self.x0, self.y1, int(round(w)), int(round(h)), res)

    def offset_of(self, other: Grid) -> tuple[int, int]:
        """(row, col) of ``other``'s upper-left pixel inside this grid (same resolution)."""
        if other.res != self.res:
            raise ValueError("grids differ in resolution")
        c, r = (other.x0 - self.x0) / self.res, (self.y1 - other.y1) / self.res
        if abs(c - round(c)) > 1e-6 or abs(r - round(r)) > 1e-6:
            raise ValueError("grids are not aligned")
        return int(round(r)), int(round(c))

    def cut(self, array: np.ndarray, other: Grid) -> np.ndarray:
        """Slice the trailing (H, W) axes of ``array`` (laid out on this grid) to ``other``."""
        r, c = self.offset_of(other)
        if r < 0 or c < 0 or r + other.height > self.height or c + other.width > self.width:
            raise ValueError("requested grid is outside this grid")
        return array[..., r:r + other.height, c:c + other.width]

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Projected x (columns) and y (rows) of pixel centres."""
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.res
        ys = self.y1 - (np.arange(self.height) + 0.5) * self.res
        return xs, ys


def union_grid(grids, res: float, snap: float) -> Grid:
    """Smallest snapped grid containing every grid in ``grids``."""
    b = np.array([g.bounds for g in grids])
    return Grid.from_bounds((b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()), res, snap)


def sample_windows(region, n: int, win_px: int, res: float, snap: float, avoid=None,
                   seed: int = 0, max_tries: int = 100_000) -> list[Grid]:
    """Draw ``n`` non-overlapping square windows inside ``region`` (a shapely geometry).

    Windows are ``win_px`` pixels wide, their corners lie on multiples of ``snap`` and they do
    not intersect ``avoid`` (e.g. labelled sites buffered by a safety distance). The draw is
    deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    side = win_px * res
    x0, y0, x1, y1 = snap_bounds(region.bounds, snap)
    nx, ny = int((x1 - x0 - side) // snap), int((y1 - y0 - side) // snap)
    if nx < 0 or ny < 0:
        raise ValueError("region is smaller than one window")
    avoid = avoid if avoid is not None else box(0, 0, 0, 0)
    out, taken = [], []
    for _ in range(max_tries):
        if len(out) == n:
            break
        wx = x0 + rng.integers(0, nx + 1) * snap
        wy = y1 - rng.integers(0, ny + 1) * snap
        w = box(wx, wy - side, wx + side, wy)
        if not region.covers(w) or w.intersects(avoid) or any(w.intersects(t) for t in taken):
            continue
        out.append(Grid(wx, wy, win_px, win_px, res))
        taken.append(w)
    if len(out) < n:
        raise RuntimeError(f"placed only {len(out)} of {n} windows; enlarge the region, reduce n "
                           "or the buffer around labelled sites")
    return out


def buffered_union(geoms, distance: float):
    """Union of geometries buffered by ``distance`` metres."""
    return unary_union([g.buffer(distance) for g in geoms])
