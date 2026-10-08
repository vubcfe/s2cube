# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Unit tests of the building blocks."""
import numpy as np
import pytest
from scipy import ndimage
from shapely.geometry import Point, box

from s2cube import geo, labels, splits, stac
from s2cube.config import Config
from s2cube.coreg import _norm, ncc_shift, ncc_surface
from s2cube.reader import read_band

from conftest import X0, Y1, write_tif


# ------------------------------------------------------------------ geo
def test_utm_crs():
    assert geo.utm_crs(107.9, 14.5).to_epsg() == 32648
    assert geo.utm_crs(-47.9, -15.8).to_epsg() == 32723


def test_grid_snapping_and_cut():
    g = geo.Grid.from_bounds((105, 207, 391, 395), 10, snap=20)
    assert g.bounds == (100, 200, 400, 400) and g.shape == (20, 30)
    sub = geo.Grid(200, 380, 5, 4, 10)
    a = np.arange(20 * 30).reshape(20, 30)
    assert np.array_equal(g.cut(a, sub), a[2:6, 10:15])
    assert g.with_res(20).shape == (10, 15)
    with pytest.raises(ValueError):
        g.cut(a, geo.Grid(395, 380, 5, 4, 10))


def test_sample_windows_reproducible_and_disjoint():
    region = box(0, 0, 5000, 5000)
    avoid = Point(2500, 2500).buffer(800)
    w1 = geo.sample_windows(region, 10, 32, 10, 20, avoid, seed=3)
    w2 = geo.sample_windows(region, 10, 32, 10, 20, avoid, seed=3)
    assert [w.bounds for w in w1] == [w.bounds for w in w2]
    for i, a in enumerate(w1):
        assert not a.geometry.intersects(avoid) and region.contains(a.geometry)
        assert a.x0 % 20 == 0 and a.y1 % 20 == 0
        for b in w1[i + 1:]:
            assert not a.geometry.intersects(b.geometry)
    with pytest.raises(RuntimeError):
        geo.sample_windows(box(0, 0, 400, 400), 50, 32, 10, 20, max_tries=2000)


# ------------------------------------------------------------------ stac
def _item(date, tile, gen, epsg=32648, baseline="05.10"):
    return {"id": f"{date}-{tile}-{gen}", "assets": {"B04": {"href": "x"}},
            "properties": {"datetime": f"{date}T03:00:00Z", "s2:mgrs_tile": tile, "proj:epsg": epsg,
                           "s2:generation_time": gen, "s2:processing_baseline": baseline}}


def test_group_by_date_keeps_latest_processing_and_orders_by_crs():
    items = [_item("2024-01-01", "48PZA", "2024-01-01T05"), _item("2024-01-01", "48PZA", "2024-03-01T05"),
             _item("2024-01-01", "49PAA", "2024-01-01T05", epsg=32649), _item("2024-01-01", "48PZB", "a")]
    g = stac.group_by_date(items, 32648)
    assert list(g) == ["2024-01-01"]
    ids = [it["id"] for it in g["2024-01-01"]]
    assert ids[0].endswith("2024-03-01T05") and ids[-1].startswith("2024-01-01-49PAA")
    assert len(ids) == 3


def test_dn_offset_from_baseline_and_earth_search_flag():
    pc = stac.get_preset("planetary-computer")
    es = stac.get_preset("earth-search")
    assert stac.item_offset(_item("2024-01-01", "T", "g", baseline="05.10"), pc) == 1000
    assert stac.item_offset(_item("2021-01-01", "T", "g", baseline="02.09"), pc) == 0
    # Earth Search: offset already removed, although raster:bands still lists offset -0.1
    it = {"assets": {"red": {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}},
          "properties": {"s2:processing_baseline": "05.11", "earthsearch:boa_offset_applied": True}}
    assert stac.item_offset(it, es) == 0
    assert es.asset_key("B8A") == "nir08" and pc.asset_key("B8A") == "B8A"


def test_proj_code_and_grid_code():
    it = {"id": "x", "properties": {"proj:code": "EPSG:32748", "grid:code": "MGRS-48MXT"}}
    assert stac.item_epsg(it) == 32748 and stac.item_tile(it) == "48MXT"


# ------------------------------------------------------------------ reader
def test_read_band_mosaic_offset_and_other_zone(tmp_path):
    a = np.full((50, 50), 2000, "uint16")
    a[:, 25:] = 0                                        # east half missing in tile 1
    write_tif(tmp_path / "t1.tif", a, 10)
    b = np.full((50, 50), 3000, "uint16")
    write_tif(tmp_path / "t2.tif", b, 10)
    g = geo.Grid(X0, Y1, 50, 50, 10)
    out = read_band([str(tmp_path / "t1.tif"), str(tmp_path / "t2.tif")], g, "EPSG:32648", offsets=[1000, 1000])
    assert (out[:, :25] == 1000).all() and (out[:, 25:] == 2000).all()
    # a tile stored in the neighbouring UTM zone is reprojected on read
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds("EPSG:32648", "EPSG:32649", *g.bounds)
    c = np.full((80, 80), 1500, "uint16")
    write_tif(tmp_path / "z49.tif", c, 10, x0=w - 100, y1=n + 100, epsg=32649)
    out = read_band([str(tmp_path / "z49.tif")], g, "EPSG:32648")
    assert (out == 1500).mean() > 0.95


# ------------------------------------------------------------------ coreg
def test_ncc_fft_equals_direct_definition():
    rng = np.random.default_rng(1)
    E = ndimage.gaussian_filter(rng.normal(size=(60, 50)), 2)
    S = ndimage.gaussian_filter(rng.normal(size=(60, 50)), 2) + np.roll(E, (2, -3), (0, 1))
    m = 5
    fast = ncc_surface(S, E, m)
    Sc = _norm(S[m:-m, m:-m])
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            direct = np.mean(Sc * _norm(E[m + dy:60 - m + dy, m + dx:50 - m + dx]))
            assert fast[dy + m, dx + m] == pytest.approx(direct, abs=1e-6)


def test_ncc_subpixel():
    rng = np.random.default_rng(0)
    E = ndimage.gaussian_filter(rng.normal(size=(140, 140)), 3)
    S = ndimage.shift(E, (-2.4, 3.6), order=3)           # S[r, c] = E[r - 2.4, c + 3.6]
    dx, dy, ncc = ncc_shift(S, E, 8)
    assert dx == pytest.approx(-3.6, abs=0.3) and dy == pytest.approx(2.4, abs=0.3) and ncc > 0.9


# ------------------------------------------------------------------ labels
def test_assign_purity_and_ignore():
    g = geo.Grid(0, 100, 10, 10, 10)
    half = box(0, 50, 100, 100)
    sliver = box(0, 0, 4, 100)                           # covers 40 % of the first column
    frac = labels.class_fractions([(half, 1), (sliver, 2)], g, [1, 2], 10)
    lab = labels.assign(frac, [1, 2], 0.7)
    assert (lab[:5, 1:] == 1).all() and (lab[5:, 1:] == 0).all()
    assert (lab[:, 0] == labels.IGNORE).all()
    ann = np.ones((10, 10), bool)
    ann[0, 5] = False
    assert labels.assign(frac, [1, 2], 0.7, ann)[0, 5] == labels.IGNORE
    with pytest.raises(ValueError):
        labels.assign(frac, [0, 300], 0.7)


def test_flag_drift():
    # 3 periods x 4 polygons; polygon 3 cleared in the last period, period 2 is globally drier
    v = np.array([[0.80, 0.75, 0.78, 0.82],
                  [0.70, 0.66, 0.69, 0.71],
                  [0.81, 0.76, 0.77, 0.45]])
    f = labels.flag_drift(v, drop=0.15, low=0.4)
    assert f[2, 3] and f.sum() == 1


def test_blocks():
    b = labels.blocks((5, 7), 3)
    assert b[0, 0] == 0 and b[0, 3] == 1 and b[3, 0] == 3 and b.max() == 5


# ------------------------------------------------------------------ splits
def test_clusters_and_folds_respect_gap():
    geoms = {"a": box(0, 0, 100, 100), "b": box(150, 0, 250, 100), "c": box(3000, 0, 3100, 100),
             "d": box(6000, 0, 6100, 100), "e": box(9000, 0, 9100, 100)}
    cl = splits.clusters(geoms, 500)
    assert ["a", "b"] in cl and len(cl) == 4
    folds = splits.make_folds(geoms, 3, 500, 500)
    for f in folds:
        assert {"a", "b"} <= set(f["test"]) or {"a", "b"} <= set(f["val"]) or {"a", "b"} <= set(f["train"])
        assert f["min_gap_m"] >= 500
    with pytest.raises(ValueError):
        splits.make_folds(geoms, 5, 500, 500)


def test_nested_subsets():
    bf = {f"s{i % 3}:{i}": (i % 10) / 10 for i in range(60)}
    sub = splits.nested_subsets(bf, ["s0", "s1"], [0.05, 0.1, 0.5, 1.0], [0, 1])
    for s in ("0", "1"):
        lst = [set(sub[s][r]) for r in ("0.05", "0.1", "0.5", "1.0")]
        assert all(a <= b for a, b in zip(lst, lst[1:]))
        assert all(k.split(":")[0] in ("s0", "s1") for k in lst[-1]) and len(lst[-1]) == 40
    assert sub["0"]["0.5"] != sub["1"]["0.5"]


# ------------------------------------------------------------------ config
def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        Config.from_dict({"bands": ["B99"]})
    with pytest.raises(ValueError):
        Config.from_dict({"sites": {"unknown": 1}})
    with pytest.raises(ValueError):
        Config.from_dict({"res": 10, "bands": ["B01"], "source": "nowhere"})
    c = Config.from_dict({"output": "x.h5"}, base_dir=tmp_path)
    assert c.path("x.h5") == tmp_path / "x.h5"


# ------------------------------------------------------------------ sites from labels, class maps
def test_tile_sites_from_label_polygons():
    from s2cube.cube import tile_sites

    polys = [box(100, 100, 900, 900), box(2050, 50, 2100, 100)]     # one big field, one tiny field
    t = tile_sites(polys, 1000, 0.2)
    assert list(t) == ["E0000p00_N0000p00"] and t["E0000p00_N0000p00"].bounds == (0, 0, 1000, 1000)
    assert len(tile_sites(polys, 1000, 0.001)) == 2
    many = [box(i * 1000 + 100, 100, i * 1000 + 900, 900) for i in range(10)]
    a, b = tile_sites(many, 1000, 0.2, max_sites=4, seed=1), tile_sites(many, 1000, 0.2, max_sites=4, seed=1)
    assert len(a) == 4 and list(a) == list(b)


def test_feature_classes_with_class_map():
    feats = [(box(0, 0, 1, 1), {"crop": "Maize"}), (box(1, 0, 2, 1), {"crop": "Cassava"}),
             (box(2, 0, 3, 1), {"crop": "Tomato"})]
    g, ign = labels.feature_classes(feats, "crop", {"Maize": 1, "Cassava": 2})
    assert [k for _, k in g] == [1, 2] and len(ign) == 1
    g, ign = labels.feature_classes(feats, "crop", {"Maize": 1}, unmapped="background")
    assert [k for _, k in g] == [1] and ign == []
    g, _ = labels.feature_classes(feats)
    assert [k for _, k in g] == [1, 1, 1]


# ------------------------------------------------------------------ fixes after external review
def test_res_60_rejected():
    with pytest.raises(ValueError):
        Config.from_dict({"res": 60, "bands": ["B01", "B09"]})


def test_read_groups_split_distant_sites():
    from s2cube.cube import read_groups

    gs = [geo.Grid(0, 1000, 10, 10, 10), geo.Grid(200, 1000, 10, 10, 10), geo.Grid(50000, 1000, 10, 10, 10)]
    groups = read_groups(gs, 10, 20, 2000)
    assert [m for _, m in groups] == [[0, 1], [2]]
    assert groups[0][0].width == 30 and groups[1][0].width == 10


def test_flag_drift_needs_population():
    # one polygon alone can never be flagged by the relative rule; within a population it is
    alone = np.array([[0.8], [0.5]])
    assert not labels.flag_drift(alone, drop=0.15, low=None).any()
    pop = np.array([[0.8, 0.8, 0.8], [0.79, 0.81, 0.5]])
    f = labels.flag_drift(pop, drop=0.15, low=None)
    assert f[1, 2] and f.sum() == 1


def test_xyz_url_must_have_placeholders(tmp_path):
    from s2cube.coreg import XYZReference

    with pytest.raises(ValueError):
        XYZReference("https://x/{z}/{y}.png", 17, "EPSG:32648", tmp_path)
