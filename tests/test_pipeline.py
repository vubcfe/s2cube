# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""End-to-end test on synthetic local scenes: plan -> create -> fetch -> coreg -> labels -> splits
-> qa -> datasets -> manifest."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from s2cube import Config, Cube, coreg, qa, stac
from s2cube.cube import create, fetch, plan
from s2cube.datasets import LabeledBlocks, UnlabeledWindows
from s2cube.labels import IGNORE, build_labels
from s2cube.splits import build_splits

from conftest import BANDS, CLOUDY, DATES, GAIN, OFFSET, SHIFTED, block_mean


@pytest.fixture()
def built(project, scenes):
    cfg = Config.load(project / "project.yaml")
    p = plan(cfg)
    by_date = stac.group_by_date(scenes["items"], p.crs.to_epsg())
    path = create(cfg, p, by_date)
    written = fetch(path, log=lambda *_: None)
    assert written == list(range(len(DATES)))
    return cfg, path, scenes


def test_plan_grids(project):
    cfg = Config.load(project / "project.yaml")
    p = plan(cfg)
    assert p.crs.to_epsg() == 32648
    assert sorted(p.sites) == ["a", "b", "c", "d"]
    for g in p.sites.values():
        assert g.x0 % 20 == 0 and g.y1 % 20 == 0          # snapped to the 20 m grid
        assert (g.width, g.height) == (40, 40)
    assert len(p.windows) == 4
    for w in p.windows:
        assert all(w.geometry.distance(s.geometry) >= 100 for s in p.sites.values())


def test_values_are_harmonised_and_exact(built):
    cfg, path, scenes = built
    with Cube(path) as c:
        assert c.dates == DATES and c.done.all()
        g = c.site_grid("a")
        X, scl = c.site("a", 0)
    # expected 10 m B04 from the 1 m texture, offset removed
    t = scenes["texture"]
    r0, c0 = int(1600000 - g.y1), int(g.x0 - 800000)
    sub = t[r0:r0 + 400, c0:c0 + 400]
    exp = (OFFSET + 300 + 3000 * GAIN["B04"] * block_mean(sub, 10)).astype("uint16") - OFFSET
    assert np.array_equal(X[BANDS.index("B04")], exp)
    # 20 m band replicated 2 x 2, not interpolated
    b11 = X[BANDS.index("B11")]
    assert np.array_equal(b11[::2, ::2].repeat(2, 0).repeat(2, 1), b11)
    assert (scl == 4).all()


def test_qa_reflects_clouds(built):
    _, path, _ = built
    rows = qa.date_table(path)
    by = {r["date"]: r for r in rows}
    assert by[CLOUDY]["clear"] < 0.75 and by[DATES[0]]["clear"] == 1.0
    s = qa.summary(path)
    assert s["n_dates"] == len(DATES) and s["n_done"] == len(DATES)


def test_coregistration_detects_and_fixes_known_shift(built):
    cfg, path, _ = built
    rows = coreg.measure(path, cfg.coreg, log=lambda *_: None)
    by = {r["date"]: r for r in rows}
    (date, (e, n)), = SHIFTED.items()
    # content was displaced by (e, n): it must move by (-e, -n)
    assert by[date]["reliable"] == 1
    assert by[date]["east_m"] == pytest.approx(-e, abs=1.0)
    assert by[date]["north_m"] == pytest.approx(-n, abs=1.0)
    for d in DATES:
        if d not in SHIFTED and by[d]["reliable"]:
            assert np.hypot(by[d]["east_m"], by[d]["north_m"]) < 1.0
    i, ref = DATES.index(date), 0

    def misfit():
        with Cube(path) as c:
            a = c.site("a", i)[0][BANDS.index("B04")].astype(float)
            b = c.site("a", ref)[0][BANDS.index("B04")].astype(float)
        return np.sqrt(np.mean((a - b)[2:-2, 2:-2] ** 2))

    before = misfit()
    fixed = coreg.apply(path, cfg.coreg, rows=rows, log=lambda *_: None)
    assert [DATES[i] for i in fixed] == [date]
    # the corrected date matches an undisplaced date far better; the remaining difference is
    # mostly the smoothing of bilinear re-reading on this deliberately aliased texture
    assert misfit() < 0.5 * before
    # the verification pass measures a small residual (bilinear re-reading smooths the image,
    # so the measured residual is somewhat larger than the true one, here ~0.55 m)
    resid = coreg.verify(path, cfg.coreg, log=lambda *_: None)
    assert np.hypot(resid[0]["east_m"], resid[0]["north_m"]) < 1.5
    with h5py.File(path, "r") as f:
        assert f["coreg/status"][DATES.index(date)] == 2
    assert Path(path).with_suffix(".orig.h5").exists()
    # a second run must not correct the same date twice
    assert coreg.apply(path, cfg.coreg, log=lambda *_: None) == []


def test_labels_splits_datasets_manifest(built, project):
    cfg, path, _ = built
    lp = build_labels(cfg, log=lambda *_: None)
    with h5py.File(lp, "r") as f:
        lab = f["a/label"][:]
        frac = f["a/frac"][:]
    assert set(np.unique(lab)) <= {0, 1, IGNORE}
    assert (lab[:20] == 1).all() and (lab[20:] == 0).all()   # northern half labelled positive
    assert frac.shape == (1, 40, 40)

    sp = build_splits(cfg, log=lambda *_: None)
    doc = json.loads(open(sp).read())
    assert len(doc["folds"]) == 3 and doc["checks"]["nested"]
    for f in doc["folds"]:
        assert not set(f["test"]) & set(f["train"]) and not set(f["val"]) & set(f["train"])
        assert f["min_gap_m"] >= 500
    assert doc["checks"]["min_window_site_gap_m"] >= 100

    ds = LabeledBlocks(path, lp, sp, fold=0, role="train", ratio=0.5, seed=0)
    X, valid, y = ds[0]
    assert X.shape == (len(DATES), len(BANDS), 10, 10) and y.shape == (10, 10) and valid.dtype == bool
    full = LabeledBlocks(path, lp, sp, fold=0, role="train", ratio=1.0, seed=0)
    assert len(ds) <= len(full)
    uw = UnlabeledWindows(path)
    assert len(uw) == 4 and uw[0][0].shape == (len(DATES), len(BANDS), 16, 16)

    m = qa.write_manifest([path, lp, sp], project / "MANIFEST.sha256")
    assert qa.check_manifest(m) == []


def test_recreate_from_frozen_scenes(built, tmp_path):
    """A second build with scenes_from uses exactly the recorded scenes and gives identical data."""
    import yaml

    from s2cube.cube import build

    cfg, path, _ = built
    d = yaml.safe_load(cfg.to_yaml())
    d.update(output=str(tmp_path / "again.h5"), scenes_from=str(path))
    for sec, key in (("sites", "path"), ("unlabeled", "aoi"), ("labels", "path")):
        d[sec][key] = str(cfg.path(d[sec][key]))
    cfg2 = Config.from_dict(d, base_dir=tmp_path)
    p2 = build(cfg2, log=lambda *_: None)
    with Cube(path) as a, Cube(p2) as b:
        assert a.dates == b.dates and a.scenes == b.scenes
        assert np.array_equal(a.site("a")[0], b.site("a")[0])


def test_refined_only_reference_selects_flagged_dates():
    from s2cube.coreg import SelfReference

    clear = np.array([1.0, 0.95, 1.0, 0.99, 0.2])
    assert SelfReference(3, 0.9).dates(clear) == [0, 2, 3]
    assert SelfReference(3, 0.9, allowed=[1, 3, 4]).dates(clear) == [3, 1]
