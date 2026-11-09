# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Sensitivity checks on a finished example: label purity, patch guard, spatial autocorrelation and
dates without a co-registration measurement.

    python side_experiments.py --project ../netherlands/project.yaml --out side.json

The purity and guard runs use the Random Forest of run_experiment.py on cube.h5, seeds 0, 1 and 2.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from run_experiment import cap_per_class, fit_rf, load_pixels, monthly_composite, scores, select

from s2cube import Config, Cube
from s2cube.labels import build_labels
from s2cube.splits import build_splits

CLEAR = (4, 5, 6)


def evaluate(data, folds, seeds, n_jobs, cap=20000):
    rows = []
    for fold in folds:
        xt, yt, _ = select(data, fold, "train")
        xs, ys, _ = select(data, fold, "test")
        for seed in seeds:
            keep = cap_per_class(yt, cap, np.random.default_rng(seed))
            predict = fit_rf(xt[keep], yt[keep], None, None, seed, n_jobs, None)
            rows.append(scores(ys, predict(xs))[0])
    return {"macro_f1_mean": round(float(np.mean(rows)), 4), "macro_f1_std": round(float(np.std(rows, ddof=1)), 4),
            "n_runs": len(rows)}


def purity_sweep(cfg, root, classes, values, seeds, n_jobs):
    folds = json.loads((root / "splits.json").read_text())["folds"]
    out = {}
    for p in values:
        c = copy.deepcopy(cfg)
        c.labels.purity = p
        path = root / f"labels_p{int(p * 100)}.h5"
        build_labels(c, out_path=path, log=lambda *_: None)
        data = load_pixels(root / "cube.h5", path, classes)
        out[str(p)] = {"n_labelled_pixels": int(sum(len(d["y"]) for d in data.values())),
                       **evaluate(data, folds, seeds, n_jobs)}
    return out


def guard_sweep(cfg, root, classes, guards, seeds, n_jobs):
    data = load_pixels(root / "cube.h5", root / "labels.h5", classes)
    out = {}
    for g in guards:
        c = copy.deepcopy(cfg)
        c.splits.strategy, c.splits.patch_guard_blocks = "patches", g
        path = root / f"splits_guard{g}.json"
        build_splits(c, labels_path=root / "labels.h5", out_path=path, log=lambda *_: None)
        folds = json.loads(path.read_text())["folds"]
        out[str(g)] = {"gap_m": folds[0]["min_gap_m"],
                       "train_blocks": int(np.mean([f["n_blocks"]["train"] for f in folds])),
                       **evaluate(data, folds, seeds, n_jobs)}
    return out


def correlogram(root, bins, month=7, band="B08", n_px=30000, n_pairs=4_000_000, seed=0):
    """Correlation of one monthly composite band and label agreement against distance between pixels."""
    rng = np.random.default_rng(seed)
    xs, ys, vs, ls = [], [], [], []
    with Cube(root / "cube.h5") as c, h5py.File(root / "labels.h5", "r") as lf:
        months = np.array([int(d[5:7]) for d in c.dates])
        b = c.band_index(band)
        for sid in c.sites:
            X, scl = c.site(sid)
            comp = monthly_composite(X, scl, months)[..., month - 1, b]
            lab = lf[f"{sid}/label"][:]
            g = c.site_grid(sid)
            rr, cc = np.nonzero((lab > 0) & (lab < 255))
            xs.append(g.x0 + (cc + 0.5) * g.res)
            ys.append(g.y1 - (rr + 0.5) * g.res)
            vs.append(comp[rr, cc])
            ls.append(lab[rr, cc])
    x, y, v, lab = (np.concatenate(a) for a in (xs, ys, vs, ls))
    pick = rng.choice(len(x), min(n_px, len(x)), replace=False)
    x, y, v, lab = x[pick], y[pick], (v[pick] - v[pick].mean()) / v[pick].std(), lab[pick]
    i, j = rng.integers(0, len(x), n_pairs), rng.integers(0, len(x), n_pairs)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() > 200:
            out.append({"from_m": lo, "to_m": hi, "n_pairs": int(m.sum()),
                        "correlation": round(float((v[i[m]] * v[j[m]]).mean()), 3),
                        "same_class": round(float((lab[i[m]] == lab[j[m]]).mean()), 3)})
    return out


def unmeasured_dates(root):
    """How much clear data sits on dates that could not be checked."""
    rows = list(csv.DictReader(open(root / "dates.csv")))
    status = np.array([int(r["coreg_status"]) for r in rows])
    with Cube(root / "cube.h5") as c:
        clear = np.zeros(len(rows))
        total = 0
        for sid in c.sites:
            _, scl = c.site(sid)
            clear += np.isin(scl, CLEAR).sum(axis=(1, 2))
            total += scl[0].size
    frac = clear / total
    refined = [r.get("esa_refined", "-1") for r in rows]
    un = status == 0
    return {"n_dates": len(rows), "n_unmeasured": int(un.sum()),
            "share_of_clear_observations_on_unmeasured": round(float(clear[un].sum() / clear.sum()), 4),
            "median_clear_fraction_unmeasured": round(float(np.median(frac[un])), 4),
            "median_clear_fraction_measured": round(float(np.median(frac[~un])), 4),
            "unmeasured_with_clear_above_0.5": int((frac[un] > 0.5).sum()),
            "not_refined_and_unmeasured": sum(1 for u, r in zip(un, refined) if u and r == "0"),
            "clear_share_on_not_refined_unmeasured": round(
                float(sum(c for c, u, r in zip(clear, un, refined) if u and r == "0") / clear.sum()), 4)}


def shift_placebo(root, ids, eta2):
    """Shift the original image of each corrected date both ways with bilinear interpolation.

    Both copies are equally smooth, so the difference between them isolates alignment from smoothing.
    """
    from scipy import ndimage

    out = []
    with Cube(root / "cube.orig.h5") as c:
        with h5py.File(root / "cube.h5", "r") as f:
            status, shift = f["coreg/status"][:], f["coreg/shift_m"][:]
        red, nir = c.band_index("B04"), c.band_index("B08")
        for t in np.nonzero(status == 2)[0]:
            east, north = shift[t] / c.res
            res = {"plus": [], "minus": [], "none": []}
            for sid in c.sites:
                X, scl = c.site(sid, int(t))
                clear = ndimage.binary_erosion(np.isin(scl, CLEAR), iterations=2)
                for name, sign in (("plus", 1), ("minus", -1), ("none", 0)):
                    v = [ndimage.shift(X[b].astype("f4"), (sign * north, -sign * east), order=1, mode="nearest")
                         for b in (nir, red)]
                    e, n = eta2((v[0] - v[1]) / (v[0] + v[1] + 1e-6), np.where(clear, ids[sid], 0))
                    if n:
                        res[name].append(e)
            out.append({"date_index": int(t), "shift_m": [round(float(x), 2) for x in shift[t]],
                        **{k: round(float(np.mean(v)), 4) for k, v in res.items()}})
    return out


def parcel_fit(root, polygons):
    """Eta squared of NDVI by parcel per date, for the corrected and the original cube.

    The parcels come from the label file, not from s2cube, so this measures how well each image
    follows independent field boundaries. Corrected dates are compared with control dates.
    """
    from rasterio.features import rasterize

    from s2cube.vectors import read_geometries

    def eta2(ndvi, ids):
        m = ids > 0
        v, g = ndvi[m], ids[m]
        if len(v) < 200:
            return np.nan, 0
        n = np.bincount(g)
        s = np.bincount(g, v)
        q = np.nonzero(n)[0]
        between = (s[q] ** 2 / n[q]).sum() - v.sum() ** 2 / len(v)
        total = ((v - v.mean()) ** 2).sum()
        return between / total, len(v)

    rows = []
    with Cube(root / "cube.h5") as c1, Cube(root / "cube.orig.h5") as c0:
        status = c1.f["coreg/status"][:]
        red, nir = c1.band_index("B04"), c1.band_index("B08")
        geoms = [g for g, _ in read_geometries(polygons, "EPSG:4326", c1.crs)]
        ids = {}
        for sid in c1.sites:
            grid = c1.site_grid(sid)
            shapes = [(g, i + 1) for i, g in enumerate(geoms) if g.intersects(grid.geometry)]
            ids[sid] = rasterize(shapes, out_shape=grid.shape, transform=grid.transform, dtype="int32")
        for t in np.nonzero(status > 0)[0]:
            for sid in c1.sites:
                vals = []
                for c in (c1, c0):
                    X, scl = c.site(sid, int(t))
                    a, b = X[nir].astype("f4"), X[red].astype("f4")
                    vals.append(((a - b) / (a + b + 1e-6), np.isin(scl, CLEAR)))
                both = vals[0][1] & vals[1][1]
                e = [eta2(v, np.where(both, ids[sid], 0)) for v, _ in vals]
                if e[0][1]:
                    rows.append((int(t), int(status[t]), e[0][0], e[1][0]))
    a = np.array(rows, "f8")
    out = {"placebo": shift_placebo(root, ids, eta2)}
    for name, st in (("corrected_dates", 2), ("control_dates", 1)):
        r = a[a[:, 1] == st]
        per_date = [(r[r[:, 0] == t, 2].mean(), r[r[:, 0] == t, 3].mean()) for t in np.unique(r[:, 0])]
        p = np.array(per_date)
        out[name] = {"n_dates": len(p), "eta2_corrected": round(float(p[:, 0].mean()), 4),
                     "eta2_original": round(float(p[:, 1].mean()), 4),
                     "mean_gain": round(float((p[:, 0] - p[:, 1]).mean()), 4),
                     "dates_improved": int((p[:, 0] > p[:, 1]).sum()),
                     "per_date_gain": [round(float(x), 4) for x in p[:, 0] - p[:, 1]]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="side.json")
    ap.add_argument("--only", nargs="+", default=["purity", "guard", "correlogram", "unmeasured"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    cfg = Config.load(a.project)
    root = Path(a.project).resolve().parent
    classes = sorted(set(cfg.labels.class_map.values()))
    out_path = Path(a.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = {"unmeasured": lambda: unmeasured_dates(root),
            "correlogram": lambda: correlogram(root, [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]),
            "purity": lambda: purity_sweep(cfg, root, classes, [0.5, 0.7, 0.9], a.seeds, a.n_jobs),
            "guard": lambda: guard_sweep(cfg, root, classes, [0, 1, 2], a.seeds, a.n_jobs),
            "parcel_fit": lambda: parcel_fit(root, cfg.path(cfg.labels.path))}
    for name in a.only:
        res[name] = jobs[name]()
        out_path.write_text(json.dumps(res, indent=1))
        print(name, json.dumps(res[name]), flush=True)


if __name__ == "__main__":
    main()
