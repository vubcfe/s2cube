# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Downstream crop classification on an s2cube dataset: corrected vs uncorrected cube, spatial vs
random split, Random Forest vs a small temporal CNN.

Run it in a finished example folder (cube.h5, cube.orig.h5, labels.h5, splits.json):

    python run_experiment.py --project ../netherlands/project.yaml --out results.csv

Needs scikit-learn and PyTorch, which s2cube itself does not depend on. Each pixel is described by
monthly medians of its clear observations (SCL 4, 5, 6), with empty months interpolated in time.
Nothing is tuned: the validation role is used only to stop the CNN early. Scores cover crop classes;
background (0) and ignored pixels (255) are left out. Boundary scores use pixels whose 3 x 3
neighbourhood contains another label value, where misregistration matters most.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from s2cube import Config, Cube
from s2cube.splits import build_splits

CLEAR = (4, 5, 6)
MONTHS = 12
FIELDS = ["cube", "split", "model", "fold", "seed", "n_train", "n_test", "n_test_boundary",
          "macro_f1", "oa", "macro_f1_boundary", "oa_boundary"]


def monthly_composite(X, scl, months):
    """(T, B, H, W) uint16 and (T, H, W) SCL -> (H, W, 12, B) float32 reflectance."""
    _, b, h, w = X.shape
    out = np.full((MONTHS, b, h, w), np.nan, "f4")
    for m in range(MONTHS):
        idx = np.nonzero(months == m + 1)[0]
        if not len(idx):
            continue
        x = X[idx].astype("f4") / 10000.0
        ok = np.isin(scl[idx], CLEAR)[:, None] & (X[idx] > 0)
        x[~ok] = np.nan
        seen = ok.any(axis=0)
        x[:, ~seen] = 0.0                       # keeps nanmedian free of all-NaN warnings
        med = np.nanmedian(x, axis=0)
        med[~seen] = np.nan
        out[m] = med
    fill_time(out.reshape(MONTHS, -1))
    return out.transpose(2, 3, 0, 1)


def date_series(X, scl, keep):
    """(T, B, H, W) uint16 -> (H, W, D, B) float32 on the dates ``keep``, cloudy values interpolated in time."""
    x = X[keep].astype("f4") / 10000.0
    x[~(np.isin(scl[keep], CLEAR)[:, None] & (X[keep] > 0))] = np.nan
    fill_time(x.reshape(len(keep), -1))
    return x.transpose(2, 3, 0, 1)


def fill_time(flat):
    """Interpolate NaNs of (time, N) in place along time; ends keep the nearest value, empty columns are 0."""
    n = flat.shape[0]
    good = ~np.isnan(flat)
    t = np.arange(n)[:, None]
    before = np.maximum.accumulate(np.where(good, t, -1), axis=0)
    after = np.minimum.accumulate(np.where(good, t, n)[::-1], axis=0)[::-1]
    col = np.arange(flat.shape[1])
    lo = np.where(before < 0, after, before).clip(0, n - 1)
    hi = np.where(after >= n, before, after).clip(0, n - 1)
    a, b = flat[lo, col], flat[hi, col]
    w = np.where(hi > lo, (t - lo) / np.maximum(hi - lo, 1), 0.0).astype(flat.dtype)
    filled = np.nan_to_num(a + w * (b - a), nan=0.0)
    flat[~good] = filled[~good]


def boundary_mask(lab):
    return ndimage.maximum_filter(lab, 3) != ndimage.minimum_filter(lab, 3)


def clear_dates(cube_path, min_clear):
    """Indices of the dates whose mean clear fraction over the sites reaches ``min_clear``."""
    with h5py.File(cube_path, "r") as f:
        return np.nonzero(np.nanmean(f["qa/site_clear"][:], axis=1) >= min_clear)[0]


def load_pixels(cube_path, labels_path, classes, dates=None):
    """Per site: features, labels, block ids and boundary flags of the crop pixels. Features are monthly
    composites, or the series of the date indices ``dates`` when given."""
    data = {}
    with Cube(cube_path) as c, h5py.File(labels_path, "r") as lf:
        months = np.array([int(d[5:7]) for d in c.dates])
        for sid in c.sites:
            lab = lf[f"{sid}/label"][:]
            keep = np.isin(lab, classes)
            if not keep.any():
                continue
            X, scl = c.site(sid)
            feats = monthly_composite(X, scl, months) if dates is None else date_series(X, scl, dates)
            data[sid] = {"x": feats[keep], "y": lab[keep].astype("i8"),
                         "block": lf[f"{sid}/block"][:][keep], "edge": boundary_mask(lab)[keep]}
    return data


def select(data, fold, role):
    """Pixels of one role of a fold: by block (patch split) or by site (spatial split)."""
    parts = []
    if "blocks" in fold:
        by_site = {}
        for key in fold["blocks"][role]:
            sid, b = key.split(":")
            by_site.setdefault(sid, set()).add(int(b))
        for sid, blocks in by_site.items():
            if sid in data:
                d = data[sid]
                m = np.isin(d["block"], list(blocks))
                parts.append((d["x"][m], d["y"][m], d["edge"][m]))
    else:
        parts = [(data[s]["x"], data[s]["y"], data[s]["edge"]) for s in fold[role] if s in data]
    if not parts:
        return np.empty((0, MONTHS, 1), "f4"), np.empty(0, "i8"), np.empty(0, bool)
    return tuple(np.concatenate(p) for p in zip(*parts))


def cap_per_class(y, n, rng):
    idx = [rng.permutation(np.nonzero(y == k)[0])[:n] for k in np.unique(y)]
    return np.concatenate(idx)


def fit_rf(x, y, xv, yv, seed, n_jobs, device):
    model = RandomForestClassifier(300, class_weight="balanced", n_jobs=n_jobs, random_state=seed)
    model.fit(x.reshape(len(x), -1), y)
    return lambda q: model.predict(q.reshape(len(q), -1))


def fit_tempcnn(x, y, xv, yv, seed, n_jobs, device, epochs=30, patience=5, batch=1024):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    classes = np.unique(y)
    mean, std = x.mean((0, 1)), x.std((0, 1)) + 1e-6

    def tensor(a):
        return torch.as_tensor(((a - mean) / std).transpose(0, 2, 1), dtype=torch.float32, device=device)

    layers, width = [], x.shape[2]
    for _ in range(3):
        layers += [nn.Conv1d(width, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3)]
        width = 64
    net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(64 * x.shape[1], len(classes))).to(device)
    count = np.array([(y == k).sum() for k in classes], "f4")
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(count.sum() / (len(count) * count), device=device))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = tensor(x)
    yt = torch.as_tensor(np.searchsorted(classes, y), device=device)

    def predict(q):
        net.eval()
        with torch.no_grad():
            parts = [net(tensor(q[i:i + 65536])).argmax(1).cpu().numpy() for i in range(0, len(q), 65536)]
        return classes[np.concatenate(parts)] if parts else np.empty(0, "i8")

    best, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        net.train()
        for i in torch.randperm(len(xt), device=device).split(batch):
            if len(i) < 2:
                continue
            opt.zero_grad()
            loss_fn(net(xt[i]), yt[i]).backward()
            opt.step()
        score = f1_score(yv, predict(xv), average="macro", labels=np.unique(yv)) if len(yv) else 0.0
        if score > best:
            best, wait = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return predict


def scores(y, p):
    if not len(y):
        return float("nan"), float("nan")
    return f1_score(y, p, average="macro", labels=np.unique(y)), accuracy_score(y, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--cubes", nargs="+", default=["cube.h5", "cube.orig.h5"])
    ap.add_argument("--splits", nargs="+", default=["spatial", "random"], choices=["spatial", "random"])
    ap.add_argument("--models", nargs="+", default=["rf", "tempcnn"], choices=["rf", "tempcnn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="default: all folds")
    ap.add_argument("--labels", default="labels.h5")
    ap.add_argument("--features", default="monthly", choices=["monthly", "dates"])
    ap.add_argument("--min-clear", type=float, default=0.5, help="--features dates: dates to keep")
    ap.add_argument("--max-per-class", type=int, default=20000)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    cfg = Config.load(a.project)
    root = Path(a.project).resolve().parent
    labels = root / a.labels
    random_path = root / "splits_random.json"
    if "random" in a.splits and not random_path.exists():
        cfg.splits.strategy, cfg.splits.patch_blocks, cfg.splits.patch_guard_blocks = "patches", 1, 0
        build_splits(cfg, labels_path=labels, out_path=random_path, log=lambda *_: None)
    folds = {"spatial": root / "splits.json", "random": random_path}
    classes = sorted(set(cfg.labels.class_map.values()))
    fit = {"rf": fit_rf, "tempcnn": fit_tempcnn}

    dates = clear_dates(root / "cube.h5", a.min_clear) if a.features == "dates" else None
    if dates is not None:
        print(f"{len(dates)} dates with a mean clear fraction of at least {a.min_clear}", flush=True)
    with open(a.out, "a", newline="") as fo:
        writer = csv.DictWriter(fo, FIELDS)
        if fo.tell() == 0:
            writer.writeheader()
        for cube in a.cubes:
            data = load_pixels(root / cube, labels, classes, dates)
            for split in a.splits:
                for fold in json.loads(folds[split].read_text())["folds"]:
                    if a.folds is not None and fold["fold"] not in a.folds:
                        continue
                    xt, yt, _ = select(data, fold, "train")
                    xv, yv, _ = select(data, fold, "val")
                    xs, ys, edge = select(data, fold, "test")
                    for seed in a.seeds:
                        rng = np.random.default_rng(seed)
                        keep = cap_per_class(yt, a.max_per_class, rng)
                        for name in a.models:
                            predict = fit[name](xt[keep], yt[keep], xv, yv, seed, a.n_jobs, a.device)
                            p = predict(xs)
                            f1, oa = scores(ys, p)
                            f1b, oab = scores(ys[edge], p[edge])
                            writer.writerow(dict(cube=cube, split=split, model=name, fold=fold["fold"], seed=seed,
                                                 n_train=len(keep), n_test=len(ys), n_test_boundary=int(edge.sum()),
                                                 macro_f1=round(f1, 4), oa=round(oa, 4),
                                                 macro_f1_boundary=round(f1b, 4), oa_boundary=round(oab, 4)))
                            fo.flush()
                            print(cube, split, name, fold["fold"], seed, f"F1 {f1:.3f} OA {oa:.3f}", flush=True)


if __name__ == "__main__":
    main()
