# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Tests against the live STAC catalogues (``pytest -m network``)."""
import numpy as np
import pytest

from s2cube import stac
from s2cube.geo import Grid
from s2cube.reader import read_date

pytestmark = pytest.mark.network

BBOX = (5.60, 52.50, 5.62, 52.52)              # Flevoland, the Netherlands (UTM 31N)
EPSG = 32631
GRID = Grid(676800.0, 5821600.0, 40, 40, 10.0)  # 400 m x 400 m inside BBOX
DATE = "2025-05-01"


@pytest.mark.parametrize("source", ["planetary-computer", "earth-search"])
def test_search_and_read_one_date(source):
    preset = stac.get_preset(source)
    items = stac.search(BBOX, DATE, DATE, source)
    by_date = stac.group_by_date(items, EPSG)
    assert DATE in by_date
    recs = [stac.provenance(it, preset, ["B04", "B08", "B11"]) for it in by_date[DATE]]
    # Planetary Computer serves original DN (offset 1000 since baseline 04.00); Earth Search
    # has removed it already
    assert all(r["offset"] == (1000 if source == "planetary-computer" else 0) for r in recs)
    X, scl = read_date(recs, GRID, f"EPSG:{EPSG}", ["B04", "B08", "B11"], stac.make_signer(preset), preset.asset_key)
    assert X.shape == (3, 40, 40) and (scl > 0).all()
    assert 0 < np.median(X[0]) < 3000 and np.median(X[1]) > np.median(X[0])   # vegetation: NIR > red


def test_both_catalogues_give_the_same_values():
    """The same ESA granule served by both catalogues must give identical harmonised values."""
    out = {}
    for source in ("planetary-computer", "earth-search"):
        preset = stac.get_preset(source)
        items = stac.group_by_date(stac.search(BBOX, DATE, DATE, source), EPSG)[DATE][:1]
        recs = [stac.provenance(it, preset, ["B04"]) for it in items]
        out[source] = read_date(recs, GRID, f"EPSG:{EPSG}", ["B04"], stac.make_signer(preset), preset.asset_key)[0]
    a, b = out.values()
    assert np.mean(a == b) > 0.99
