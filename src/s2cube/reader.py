# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Read one acquisition date (all MGRS tiles of one orbit pass) onto a target grid.

Each band is read at its native resolution through a GDAL warped VRT whose output grid is
exactly the target grid (optionally shifted for co-registration), then 20 m / 60 m bands
are replicated onto the 10 m grid by integer repetition, never interpolated. Tiles of the
same date are mosaicked first-valid-wins. Digital numbers are harmonised across ESA
processing baselines so that reflectance = DN / 10000 everywhere.
"""
from __future__ import annotations

import time

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

from .geo import Grid
from .stac import NATIVE_RES

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE")

#: SCL classes treated as clear land/water observations by default:
#: 4 vegetation, 5 not vegetated (bare soil), 6 water.
CLEAR_SCL = (4, 5, 6)


def coarsest_res(bands) -> int:
    """Largest native resolution among ``bands`` and SCL: the snapping step for grids."""
    return max(NATIVE_RES[b] for b in (*bands, "SCL"))


def read_band(hrefs, grid: Grid, crs, shift=(0.0, 0.0), resampling=Resampling.nearest,
              erode_px: int = 1, retries: int = 4, offsets=None) -> np.ndarray:
    """Mosaic one band from several files onto ``grid`` (pixel size = the band's native size).

    ``offsets`` (one per file) are subtracted from valid pixels before mosaicking, so tiles
    from different processing baselines are harmonised individually; valid pixels at or
    below the offset (reflectance <= 0) become 1, keeping 0 reserved for no data.

    ``shift`` = (east, north) metres: the image content is moved by this amount, i.e. pixel
    (x, y) of the output takes the source value at (x - east, y - north). When shifting, each
    tile's valid area is eroded by ``erode_px`` native pixels (bilinear support) so interpolation
    never mixes in no-data.
    """
    e, n = shift
    tf = rasterio.transform.Affine(grid.res, 0, grid.x0 - e, 0, -grid.res, grid.y1 - n)
    out = np.zeros(grid.shape, np.uint16)
    crs = CRS.from_user_input(crs)
    offsets = offsets or [0] * len(hrefs)
    for href, off in zip(hrefs, offsets):
        for attempt in range(retries):
            try:
                with rasterio.open(href) as src, WarpedVRT(
                        src, crs=crs, transform=tf, width=grid.width, height=grid.height,
                        resampling=resampling, src_nodata=0, nodata=0) as vrt:
                    a = vrt.read(1)
                break
            except rasterio.errors.RasterioIOError:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * 2 ** attempt)
        ok = a != 0
        if off:
            a = np.where(a > off, a - off, 1).astype(np.uint16)
        if (e, n) != (0.0, 0.0) and erode_px:
            ok = ndimage.binary_erosion(ok, iterations=erode_px, border_value=1)
        m = (out == 0) & ok
        out[m] = a[m]
        if not (out == 0).any():
            break
    return out


def read_date(records, grid: Grid, crs, bands, signer=lambda h: h, asset_key=lambda b: b,
              shift=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Read all ``bands`` and SCL for one date.

    ``records`` are provenance dicts (see :func:`s2cube.stac.provenance`) of the tiles of one
    date. Returns ``X`` (bands, H, W) uint16 harmonised DN (reflectance x 10000, 0 = no data)
    and ``SCL`` (H, W) uint8 (0 = no data). Pixels without SCL are set to no data in ``X``.
    """
    shifted = tuple(shift) != (0.0, 0.0)
    rs = Resampling.bilinear if shifted else Resampling.nearest
    X = np.zeros((len(bands), *grid.shape), np.uint16)
    offsets = [int(rec.get("offset", 0)) for rec in records]
    with rasterio.Env(**GDAL_ENV):
        for i, b in enumerate(bands):
            r = NATIVE_RES[b]
            hrefs = [signer(rec["assets"][asset_key(b)]) for rec in records]
            k = int(r // grid.res)
            a = read_band(hrefs, grid.with_res(r), crs, shift, rs, offsets=offsets)
            X[i] = a.repeat(k, 0).repeat(k, 1) if k > 1 else a
        k = int(NATIVE_RES["SCL"] // grid.res)
        hrefs = [signer(rec["assets"][asset_key("SCL")]) for rec in records]
        scl = read_band(hrefs, grid.with_res(NATIVE_RES["SCL"]), crs, shift, Resampling.nearest)
        scl = (scl.repeat(k, 0).repeat(k, 1) if k > 1 else scl).astype(np.uint8)
    X[:, scl == 0] = 0
    return X, scl
