# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Pixel labels from polygons, with explicit "ignore" pixels and optional per-period
checks of label validity.

Polygons are rasterised on a sub-pixel grid (``supersample`` x ``supersample`` per pixel) to
obtain, for every class, the fraction of each pixel it covers. A pixel receives a class only
if that class (or the background) covers at least ``purity`` of it; mixed pixels and pixels
outside the annotated extent get the ignore value 255. Labels drawn once are often reused for
several years although land use changes; :func:`flag_drift` compares, per polygon and per
period, a spectral index from the cube against the population of polygons and marks
polygon-periods that deviate, so that those pixels are ignored only in the affected period.
"""
from __future__ import annotations

import json
import math
import warnings

import h5py
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.ops import unary_union

from ._log import log_print
from .config import Config
from .cube import Cube
from .geo import Grid
from .vectors import read_geometries

IGNORE = 255


def cover_fraction(polygons, grid: Grid, supersample: int = 10) -> np.ndarray:
    """(H, W) float32 fraction of each pixel covered by the union of ``polygons``."""
    if not polygons:
        return np.zeros(grid.shape, "f4")
    s = supersample
    tf = Affine(grid.res / s, 0, grid.x0, 0, -grid.res / s, grid.y1)
    a = rasterize([(p, 1) for p in polygons], out_shape=(grid.height * s, grid.width * s),
                  transform=tf, dtype="uint8")
    return a.reshape(grid.height, s, grid.width, s).mean((1, 3)).astype("f4")


def class_fractions(features, grid: Grid, classes, supersample: int = 10) -> np.ndarray:
    """(C, H, W) cover fraction of every class in ``classes``; ``features`` = [(geom, class)]."""
    return np.stack([cover_fraction([g for g, c in features if c == k], grid, supersample) for k in classes])


def assign(frac: np.ndarray, classes, purity: float, annotated: np.ndarray | None = None) -> np.ndarray:
    """Hard labels: class value where one class covers >= purity, 0 where background does,
    IGNORE elsewhere, outside the annotated area, and where polygons of different classes
    overlap (conflicting annotation)."""
    if any(not 1 <= int(k) <= 254 for k in classes):
        raise ValueError("class values must be integers in 1..254 (0 = background, 255 = ignore)")
    bg = np.clip(1 - frac.sum(0), 0, 1)
    full = np.concatenate([bg[None], frac])
    vals = np.array([0, *classes], np.uint8)
    best = full.argmax(0)
    lab = np.where(full.max(0) >= purity - 1e-6, vals[best], IGNORE).astype(np.uint8)
    lab[frac.sum(0) > 1 + 1e-6] = IGNORE
    if annotated is not None:
        lab[~annotated] = IGNORE
    return lab


def blocks(shape, block_px: int) -> np.ndarray:
    """(H, W) int32 block id of each pixel for square blocks of ``block_px`` pixels."""
    h, w = shape
    bw = math.ceil(w / block_px)
    rr, cc = np.mgrid[0:h, 0:w]
    return ((rr // block_px) * bw + cc // block_px).astype(np.int32)


# ---------------------------------------------------------------- label drift
def polygon_index(cube: Cube, sid: str, polygons, periods: dict, index=("B08", "B04"),
                  clear_scl=(4, 5), min_clear: float = 0.5, min_px: int = 4, return_counts: bool = False):
    """(P, Q) median normalised-difference index per period (rows) and polygon (columns).

    ``periods`` maps a period name to the list of date indices it contains. Per date, only
    clear land pixels count (SCL 4 vegetation and 5 bare soil; water, class 6, is excluded on
    purpose because its NDVI would pull a polygon's median down), and dates where less than
    ``min_clear`` of the site is clear are skipped. Polygons covering fewer than ``min_px`` pixel centres give NaN.
    """
    g = cube.site_grid(sid)
    ids = rasterize([(p, i + 1) for i, p in enumerate(polygons)], out_shape=g.shape, transform=g.transform,
                    dtype="int32") if polygons else np.zeros(g.shape, "int32")
    a, b = cube.band_index(index[0]), cube.band_index(index[1])
    out = np.full((len(periods), len(polygons)), np.nan, "f4")
    counts = np.array([(ids == q + 1).sum() for q in range(len(polygons))], int)
    for pi, idx in enumerate(periods.values()):
        stack = []
        for t in idx:
            X, scl = cube.site(sid, t)
            clear = np.isin(scl, clear_scl)
            if clear.mean() < min_clear:
                continue
            na, nb = X[a].astype("f4"), X[b].astype("f4")
            v = (na - nb) / (na + nb + 1e-6)
            v[~clear] = np.nan
            stack.append(v)
        if not stack:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.stack(stack), 0)
            for q in range(len(polygons)):
                m = ids == q + 1
                if m.sum() >= min_px:
                    out[pi, q] = np.nanmedian(med[m])
    return (out, counts) if return_counts else out


def flag_drift(values: np.ndarray, drop: float = 0.15, low: float | None = 0.4) -> np.ndarray:
    """(P, Q) bool: polygon q is unreliable in period p.

    ``values`` stacks :func:`polygon_index` of all polygons (columns) over periods (rows).
    Each period is centred on its median over all polygons (removing weather and season
    effects common to the region); a polygon-period is flagged when it is more than ``drop``
    below the same polygon's best period, or when its raw value is below ``low``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        anom = values - np.nanmedian(values, axis=1, keepdims=True)
        best = np.nanmax(anom, axis=0, keepdims=True)
    flag = (best - anom) > drop
    if low is not None:
        flag |= values < low
    return flag & ~np.isnan(values)


def periods_by_year(dates, months=None) -> dict:
    """{year: [date indices]} optionally restricted to calendar ``months``."""
    out: dict = {}
    for i, d in enumerate(dates):
        if months is None or int(d[5:7]) in months:
            out.setdefault(d[:4], []).append(i)
    return out


# ---------------------------------------------------------------- build
def feature_classes(features, class_field=None, class_map=None, unmapped="ignore"):
    """Split features into ([(geom, class)], [geoms to ignore]).

    Without ``class_field`` every polygon is class 1. With ``class_map`` the attribute value is
    looked up (as text); polygons with other values are returned for ignoring, or dropped
    (treated as background) when ``unmapped == "background"``. Otherwise the attribute must
    be an integer class."""
    out, ignore = [], []
    cmap = {str(k): int(v) for k, v in (class_map or {}).items()}
    for g, p in features:
        if not class_field:
            out.append((g, 1))
        elif cmap:
            v = cmap.get(str(p.get(class_field)))
            if v is not None:
                out.append((g, v))
            elif unmapped == "ignore":
                ignore.append(g)
        else:
            out.append((g, int(p[class_field])))
    return out, ignore


def build_labels(cfg: Config, cube_path=None, out_path=None, log=log_print) -> str:
    """Write ``labels.h5`` next to the cube: per site the class fractions, hard labels,
    optional per-period labels (label drift) and block ids."""
    lc = cfg.labels
    if lc.path is None:
        raise ValueError("labels.path is required")
    cube_path = cube_path or cfg.path(cfg.output)
    out_path = str(out_path or cfg.path(cfg.output).with_name("labels.h5"))
    with Cube(cube_path) as c:
        crs = c.crs
        feats = read_geometries(cfg.path(lc.path), lc.crs, crs)
        geoms, unmapped = feature_classes(feats, lc.class_field, lc.class_map, lc.unmapped)
        classes = sorted(set(lc.class_map.values())) if lc.class_map else sorted({k for _, k in geoms})
        annotated_geom = None
        if lc.annotated == "labels":
            annotated_geom = unary_union([g for g, _ in feats])
        elif lc.annotated != "sites":
            annotated_geom = unary_union([g for g, _ in read_geometries(cfg.path(lc.annotated), lc.crs, crs)])
        local_idx = {sid: [q for q, (p, _) in enumerate(geoms) if p.intersects(c.site_grid(sid).geometry)]
                     for sid in c.sites}

        # label drift: one population of polygons over all sites, so that every year is centred
        # on the regional median and each polygon is compared with its own best year
        periods = periods_by_year(c.dates, lc.drift_months) if lc.drift else None
        flags_all = None
        if periods:
            vals_all = np.full((len(periods), len(geoms)), np.nan, "f4")
            best_px = np.zeros(len(geoms), int)
            for sid, idx in local_idx.items():
                if not idx:
                    continue
                v, cnt = polygon_index(c, sid, [geoms[q][0] for q in idx], periods, return_counts=True)
                for j, q in enumerate(idx):      # polygon crossing sites: keep the site where it is largest
                    if cnt[j] > best_px[q]:
                        best_px[q], vals_all[:, q] = cnt[j], v[:, j]
            flags_all = flag_drift(vals_all, lc.drift_drop, lc.drift_low)

        stats, block_frac, block_lab, cover = {}, {}, {}, {}
        with h5py.File(out_path, "w") as o:
            o.attrs.update(classes=classes, purity=lc.purity, supersample=lc.supersample, ignore=IGNORE,
                           block_px=lc.block_px, source=str(lc.path), crs=crs.to_string(),
                           annotated=lc.annotated, class_map=json.dumps(lc.class_map or {}))
            if periods:
                o.attrs.update(periods=list(periods), drift_drop=lc.drift_drop,
                               drift_low=-1 if lc.drift_low is None else lc.drift_low)
                o["drift_index"], o["drift_flag"] = vals_all, flags_all
            for sid in c.sites:
                g = c.site_grid(sid)
                idx = local_idx[sid]
                local = [geoms[q] for q in idx]
                frac = class_fractions(local, g, classes, lc.supersample)
                if annotated_geom is None:
                    annotated = cover_fraction([c.site_footprint(sid)], g, lc.supersample) >= 1 - 1e-6
                elif lc.annotated == "labels":
                    annotated = cover_fraction([annotated_geom], g, lc.supersample) >= lc.purity - 1e-6
                else:
                    annotated = cover_fraction([annotated_geom], g, lc.supersample) >= 1 - 1e-6
                lab = assign(frac, classes, lc.purity, annotated)
                other = [p for p in unmapped if p.intersects(g.geometry)]
                if other:
                    lab[cover_fraction(other, g, lc.supersample) >= 1 - lc.purity] = IGNORE
                cover[sid] = float(np.clip(frac.sum(0) + (cover_fraction(other, g, lc.supersample) if other else 0),
                                           0, 1).mean())
                grp = o.create_group(sid)
                grp["frac"], grp["label"] = frac, lab
                blk = blocks(g.shape, lc.block_px)
                grp["block"] = blk
                if periods:
                    per = np.repeat(lab[None], len(periods), 0)
                    for pi in range(len(periods)):
                        bad = [geoms[q][0] for q in idx if flags_all[pi, q]]
                        per[pi][cover_fraction(bad, g, lc.supersample) >= 1 - lc.purity] = IGNORE
                    grp["label_period"] = per
                    grp["polygon_ids"] = np.array(idx, "i4")
                for b in np.unique(blk):
                    m = (blk == b) & (lab != IGNORE)
                    if m.any():
                        block_frac[f"{sid}:{b}"] = float((lab[m] != 0).mean())
                        block_lab[f"{sid}:{b}"] = float(m.sum() / (blk == b).sum())
                stats[sid] = {str(int(k)): int((lab == k).sum()) for k in [0, *classes, IGNORE]}
            o.attrs["block_positive_fraction"] = json.dumps(block_frac)
            o.attrs["block_labelled_fraction"] = json.dumps(block_lab)
            o.attrs["pixel_counts"] = json.dumps(stats)
            o.attrs["polygon_cover"] = json.dumps(cover)
    tot = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    log(f"labels for {len(stats)} site(s): " + ", ".join(f"class {k}: {v}" for k, v in tot.items())
        + f"  -> {out_path}")
    if periods:
        log(f"label drift: {int(flags_all.any(0).sum())} of {len(geoms)} polygon(s) ignored in at least one period")
    if lc.annotated == "sites":
        low = sorted(s for s, v in cover.items() if v < 0.5)
        if low:
            log(f"WARNING: {len(low)} site(s) are less than 50 % covered by label polygons, but "
                "labels.annotated = 'sites' treats all other land as background (0). For sparse "
                "labels use labels.annotated: labels. Sites: "
                + ", ".join(low[:10]) + (" ..." if len(low) > 10 else ""))
    return out_path
