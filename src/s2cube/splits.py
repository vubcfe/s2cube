# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Spatially separated cross-validation folds and nested label-budget subsets.

Random pixel- or patch-level splits of remote-sensing data leak information, because nearby
pixels (and the same field seen in neighbouring sites) are strongly autocorrelated. Here,
sites closer than ``link_m`` are merged into clusters (single linkage on edge-to-edge
distance), whole clusters are distributed over ``k`` folds, and each fold is checked to keep
test and validation sites at least ``min_gap_m`` away from every training site.

For label-efficiency experiments, training blocks are ordered once per seed by stratified
shuffling on their positive fraction; every budget is a prefix of that order, so the 5 %
subset is contained in the 10 % subset and so on (nested subsets), which removes sampling
noise from comparisons across budgets.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np

from ._log import log_print
from ._version import __version__
from .config import Config
from .cube import Cube, site_boxes


def clusters(geoms: dict, link_m: float) -> list[list[str]]:
    """Connected components of sites whose footprints are closer than ``link_m``."""
    ids = sorted(geoms)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if geoms[ids[a]].distance(geoms[ids[b]]) < link_m:
                parent[find(ids[a])] = find(ids[b])
    comp: dict = {}
    for i in ids:
        comp.setdefault(find(i), []).append(i)
    return sorted((sorted(c) for c in comp.values()), key=lambda c: c[0])


def assign_folds(groups: list[list[str]], k: int, weights: dict | None = None) -> list[list[str]]:
    """Distribute clusters over ``k`` folds, balancing total weight (greedy, deterministic)."""
    if len(groups) < k:
        raise ValueError(f"only {len(groups)} spatial cluster(s) for k={k} folds; lower splits.k "
                         "or splits.link_m, or add sites")
    w = weights or {}
    order = sorted(groups, key=lambda c: (-sum(w.get(s, 1.0) for s in c), c[0]))
    folds: list[list[str]] = [[] for _ in range(k)]
    load = [0.0] * k
    for c in order:
        j = min(range(k), key=lambda i: (load[i], len(folds[i]), i))
        folds[j] += c
        load[j] += sum(w.get(s, 1.0) for s in c)
    return [sorted(f) for f in folds]


def make_folds(geoms: dict, k: int, link_m: float, min_gap_m: float, weights=None) -> list[dict]:
    """k folds; fold i tests on group i, validates on group i+1 and trains on the rest."""
    groups = assign_folds(clusters(geoms, link_m), k, weights)
    out = []
    for i in range(k):
        test, val = groups[i], groups[(i + 1) % k]
        train = sorted(s for j, g in enumerate(groups) if j not in (i, (i + 1) % k) for s in g)
        gap = min(geoms[a].distance(geoms[b]) for a in test + val for b in train)
        if gap < min_gap_m:
            raise ValueError(f"fold {i}: test/validation sites are only {gap:.0f} m from training sites "
                             f"(< min_gap_m = {min_gap_m}); increase splits.link_m")
        out.append({"fold": i, "test": test, "val": val, "train": train, "min_gap_m": round(float(gap), 1)})
    return out


def stratified_order(items: list[str], values: np.ndarray, rng, strata: int = 4) -> list[str]:
    """Shuffle within quantile strata of ``values`` and interleave the strata."""
    if not items:
        return []
    q = np.quantile(values, np.linspace(0, 1, strata + 1)[1:-1])
    bins = [[it for it, v in zip(items, values) if np.searchsorted(q, v, side="right") == s]
            for s in range(strata)]
    for b in bins:
        rng.shuffle(b)
    order, i = [], 0
    while any(bins):
        if bins[i % strata]:
            order.append(bins[i % strata].pop())
        i += 1
    return order


def nested_subsets(block_frac: dict, train_sites, ratios, seeds, strata: int = 4, eligible=None) -> dict:
    """{seed: {ratio: [block ids]}} of nested label budgets drawn from training sites (only blocks in
    ``eligible`` when given)."""
    train = set(train_sites)
    items = sorted(b for b in block_frac if b.split(":")[0] in train and (eligible is None or b in eligible))
    vals = np.array([block_frac[b] for b in items])
    out = {}
    for s in seeds:
        order = stratified_order(list(items), vals, np.random.default_rng(s), strata)
        out[str(s)] = {str(r): sorted(order[:max(1, math.ceil(r * len(order)))]) for r in sorted(ratios)}
    return out


def build_splits(cfg: Config, cube_path=None, labels_path=None, out_path=None, log=log_print) -> str:
    """Write ``splits.json`` with folds, nested label budgets and leakage checks."""
    sc = cfg.splits
    cube_path = cube_path or cfg.path(cfg.output)
    labels_path = labels_path or cfg.path(cfg.output).with_name("labels.h5")
    out_path = str(out_path or cfg.path(cfg.output).with_name("splits.json"))
    with Cube(cube_path) as c:
        geoms = site_boxes(c)
        windows = [w.geometry for w in c.windows]
    block_frac, weights, eligible = {}, None, None
    if labels_path and labels_path.exists():
        import h5py

        with h5py.File(labels_path, "r") as f:
            block_frac = json.loads(f.attrs["block_positive_fraction"])
            counts = json.loads(f.attrs["pixel_counts"])
            lab = json.loads(f.attrs.get("block_labelled_fraction", "{}"))
        weights = {s: float(sum(v for k, v in counts[s].items() if k != "255")) for s in counts}
        eligible = {b for b, v in lab.items() if v >= cfg.labels.block_min_labelled} if lab else None
    folds = make_folds(geoms, sc.k, sc.link_m, sc.min_gap_m, weights)
    subsets = {str(f["fold"]): nested_subsets(block_frac, f["train"], sc.ratios, sc.seeds, sc.strata, eligible)
               for f in folds} if block_frac else {}
    if block_frac:
        empty = [f["fold"] for f in folds if not subsets[str(f["fold"])][str(sc.seeds[0])][str(max(sc.ratios))]]
        if empty:
            raise ValueError(f"fold(s) {empty} have no training block with at least "
                             f"{cfg.labels.block_min_labelled:.0%} labelled pixels; lower labels.block_min_labelled")
        for f in folds:
            f["n_blocks"] = {r: sum(1 for b in block_frac if b.split(":")[0] in f[r]) for r in ("train", "val", "test")}
    win_gap = min((w.distance(g) for w in windows for g in geoms.values()), default=None)
    doc = {"s2cube_version": __version__, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "k": sc.k, "link_m": sc.link_m, "min_gap_m": sc.min_gap_m, "ratios": sorted(sc.ratios),
           "seeds": sc.seeds, "block_px": cfg.labels.block_px, "block_min_labelled": cfg.labels.block_min_labelled,
           "folds": folds, "subsets": subsets,
           "checks": {"min_test_train_gap_m": min(f["min_gap_m"] for f in folds),
                      "min_window_site_gap_m": None if win_gap is None else round(float(win_gap), 1),
                      "nested": all(set(a) <= set(b) for fs in subsets.values() for ss in fs.values()
                                    for a, b in zip(list(ss.values()), list(ss.values())[1:]))}}
    with open(out_path, "w") as fo:
        json.dump(doc, fo, indent=1)
    for f in folds:
        log(f"  fold {f['fold']}: test {f['test']}  val {f['val']}  train {len(f['train'])} site(s)  "
            f"gap >= {f['min_gap_m']:.0f} m")
    log(f"-> {out_path}")
    return out_path
