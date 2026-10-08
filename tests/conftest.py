# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Synthetic Sentinel-2-like scenes written as local GeoTIFFs, so that the whole pipeline can be
tested offline. A smooth random texture is defined on a 1 m grid; 10 m and 20 m bands are its
block means, so every resolution sees the same landscape. One date is displaced by a known
offset to test co-registration."""
from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import Affine
from scipy import ndimage

EPSG = 32648
X0, Y1 = 800000.0, 1600000.0          # upper-left of the synthetic scene (m)
SIZE = 2400                            # scene side (m)
MARGIN = 200                           # sites/windows stay inside [MARGIN, SIZE - MARGIN]
BANDS = ["B02", "B03", "B04", "B08", "B11"]
RES = {"B02": 10, "B03": 10, "B04": 10, "B08": 10, "B11": 20, "SCL": 20}
GAIN = {"B02": 0.6, "B03": 0.8, "B04": 0.7, "B08": 1.6, "B11": 1.2}
DATES = ["2024-01-05", "2024-01-15", "2024-01-25", "2024-02-04", "2024-02-14", "2024-02-24"]
SHIFTED = {"2024-02-04": (-7.0, 5.0)}  # image content displaced by (east, north) metres
CLOUDY = "2024-01-25"                  # western half covered by cloud (SCL 9)
OFFSET = 1000                          # processing baseline >= 04.00


def texture(seed=1):
    rng = np.random.default_rng(seed)
    a = ndimage.gaussian_filter(rng.normal(size=(SIZE, SIZE)), 12)
    a += 0.5 * ndimage.gaussian_filter(rng.normal(size=(SIZE, SIZE)), 4)
    a = (a - a.min()) / (a.max() - a.min())
    return a


def write_tif(path, arr, res, x0=X0, y1=Y1, epsg=EPSG):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype=arr.dtype, crs=f"EPSG:{epsg}", transform=Affine(res, 0, x0, 0, -res, y1),
                       nodata=0, tiled=True, blockxsize=64, blockysize=64) as d:
        d.write(arr, 1)


def block_mean(a, k):
    h, w = a.shape[0] // k, a.shape[1] // k
    return a[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3))


@pytest.fixture(scope="session")
def scenes(tmp_path_factory):
    """{date: [fake STAC item]} with assets on local disk, and the 1 m texture."""
    d = tmp_path_factory.mktemp("scenes")
    base = texture()
    items = []
    for date in DATES:
        e, n = SHIFTED.get(date, (0.0, 0.0))
        # content displaced by (e, n): value at (x, y) comes from (x - e, y - n)
        t = np.roll(base, (-int(n), int(e)), axis=(0, 1))
        assets = {}
        for b in BANDS:
            k = RES[b]
            dn = (OFFSET + 300 + 3000 * GAIN[b] * block_mean(t, k)).astype("uint16")
            p = d / f"{date}_{b}.tif"
            write_tif(p, dn, k)
            assets[b] = {"href": str(p)}
        scl = np.full((SIZE // 20, SIZE // 20), 4, "uint8")
        if date == CLOUDY:
            scl[:, : SIZE // 40] = 9
        p = d / f"{date}_SCL.tif"
        write_tif(p, scl, 20)
        assets["SCL"] = {"href": str(p)}
        items.append({"id": f"S2X_{date}", "assets": assets,
                      "properties": {"datetime": f"{date}T03:20:00Z", "platform": "Sentinel-2A",
                                     "proj:epsg": EPSG, "s2:mgrs_tile": "48PZZ", "eo:cloud_cover": 10.0,
                                     "s2:processing_baseline": "05.10",
                                     "s2:generation_time": f"{date}T06:00:00Z"}})
    return {"dir": d, "items": items, "texture": base}


def site_box(x, y, s=400):
    return [[x, y], [x + s, y], [x + s, y - s], [x, y - s], [x, y]]


SITES = {"a": (X0 + 260, Y1 - 260), "b": (X0 + 1700, Y1 - 260),
         "c": (X0 + 260, Y1 - 1700), "d": (X0 + 1700, Y1 - 1700)}


@pytest.fixture()
def project(tmp_path, scenes):
    """A project folder: sites, AOI, label polygons and a configuration file."""
    feats = [{"type": "Feature", "properties": {"id": k},
              "geometry": {"type": "Polygon", "coordinates": [site_box(*xy)]}} for k, xy in SITES.items()]
    (tmp_path / "sites.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    aoi = [[X0 + MARGIN, Y1 - MARGIN], [X0 + SIZE - MARGIN, Y1 - MARGIN],
           [X0 + SIZE - MARGIN, Y1 - SIZE + MARGIN], [X0 + MARGIN, Y1 - SIZE + MARGIN], [X0 + MARGIN, Y1 - MARGIN]]
    (tmp_path / "aoi.geojson").write_text(json.dumps({"type": "Feature", "properties": {},
                                                      "geometry": {"type": "Polygon", "coordinates": [aoi]}}))
    # label polygons: the northern 200 m of every site, plus a sliver giving mixed pixels
    lab = []
    for k, (x, y) in SITES.items():
        lab.append({"type": "Feature", "properties": {"id": k},
                    "geometry": {"type": "Polygon", "coordinates": [[[x, y], [x + 400, y], [x + 400, y - 200],
                                                                     [x, y - 200], [x, y]]]}})
    (tmp_path / "labels.geojsonl").write_text("\n".join(json.dumps(f) for f in lab))
    cfg = {"name": "synthetic", "start": "2024-01-01", "end": "2024-03-01", "bands": BANDS,
           "crs": f"EPSG:{EPSG}", "output": "cube.h5", "workers": 2,
           "sites": {"path": "sites.geojson", "crs": f"EPSG:{EPSG}"},
           "unlabeled": {"n": 4, "win_px": 16, "buffer_m": 100, "aoi": "aoi.geojson", "aoi_crs": f"EPSG:{EPSG}"},
           "labels": {"path": "labels.geojsonl", "crs": f"EPSG:{EPSG}", "block_px": 10},
           "splits": {"k": 3, "link_m": 500, "min_gap_m": 500, "ratios": [0.25, 0.5, 1.0], "seeds": [0, 1]},
           "coreg": {"reference": "self", "min_sites": 3, "self_dates": 5, "max_shift_m": 20}}
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path
