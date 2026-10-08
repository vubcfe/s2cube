# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Plan and build the time-series cube (HDF5), and read it back.

Building is split in two steps so that a dataset is reproducible and resumable:

1. :func:`plan` resolves the grids of labelled sites and unlabelled windows and searches
   the STAC catalogue. :func:`create` freezes the result (dates, and per date the exact
   scenes and asset URLs) into the HDF5 file.
2. :func:`fetch` downloads every date not yet marked ``done``. It never searches again, so
   an interrupted build resumes on exactly the same scenes, and a later re-processing of
   the archive cannot silently change an existing dataset.

HDF5 layout (T dates, B bands, S sites, N windows)::

    /dates                (T,)  S10        acquisition dates, ascending
    /done                 (T,)  bool       date fully written
    /scenes               (T,)  str        JSON list of scene provenance records per date
    /platform, /cloud_cover, /n_tiles   (T,)
    /sites/<id>/X         (T, B, H, W) uint16   reflectance x 10000, 0 = no data
    /sites/<id>/SCL       (T, H, W)    uint8    scene classification, 0 = no data
    /unlabeled/X          (N, T, B, w, w) uint16
    /unlabeled/SCL        (N, T, w, w)    uint8
    /unlabeled/origin     (N, 2)          upper-left corner (x, y) of each window
    /qa/site_valid, /qa/site_clear       (T, S) float32
    /qa/window_clear                     (T, N) float32
    /coreg/...                           written by s2cube.coreg
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from . import stac
from ._log import log_print
from ._version import __version__
from .config import Config
from .geo import Grid, buffered_union, sample_windows, to_crs, union_grid, utm_crs
from .reader import CLEAR_SCL, coarsest_res, read_date
from .vectors import read_features, site_footprints


@dataclass
class Plan:
    crs: CRS
    sites: dict          # id -> Grid
    footprints: dict     # id -> shapely polygon (projected)
    windows: list        # [Grid]
    region: Grid
    bbox_lonlat: tuple


def resolve_crs(cfg: Config, features_lonlat) -> CRS:
    if cfg.crs != "auto":
        return CRS.from_user_input(cfg.crs)
    c = unary_union([g for g, _ in features_lonlat]).centroid
    return utm_crs(c.x, c.y)


def tile_sites(polygons, tile_m: float, min_cover: float, max_sites=None, seed: int = 0) -> dict:
    """Square sites on a ``tile_m`` grid covering labelled polygons: {id: footprint}.

    A tile is kept if at least ``min_cover`` of its area is labelled. Ids encode the tile's
    lower-left corner in km (``E0812_N1598``)."""
    u = unary_union(polygons)
    x0, y0, x1, y1 = u.bounds
    out = {}
    for i in range(int(x0 // tile_m), int(x1 // tile_m) + 1):
        for j in range(int(y0 // tile_m), int(y1 // tile_m) + 1):
            t = box(i * tile_m, j * tile_m, (i + 1) * tile_m, (j + 1) * tile_m)
            if t.intersects(u) and t.intersection(u).area >= min_cover * t.area:
                out[f"E{i * tile_m / 1000:07.2f}_N{j * tile_m / 1000:07.2f}".replace(".", "p")] = t
    if max_sites is not None and len(out) > max_sites:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(sorted(out), max_sites, replace=False))
        out = {k: out[k] for k in keep}
    return out


def plan(cfg: Config) -> Plan:
    """Resolve grids for sites, unlabelled windows and the region read on every date."""
    sc = cfg.sites
    if sc.from_labels:
        raw = read_features(cfg.path(cfg.labels.path))
        src_crs = cfg.labels.crs
    elif sc.path:
        raw = read_features(cfg.path(sc.path))
        src_crs = sc.crs
    else:
        raise ValueError("set sites.path or sites.from_labels")
    if not raw:
        raise ValueError("no features to derive sites from")
    lonlat = [(to_crs(g, src_crs, "EPSG:4326"), p) for g, p in raw]
    crs = resolve_crs(cfg, lonlat)
    feats = [(to_crs(g, src_crs, crs), p) for g, p in raw]
    if sc.from_labels:
        fps = tile_sites([g for g, _ in feats], sc.tile_m, sc.min_cover, sc.max_sites, sc.seed)
        if not fps:
            raise ValueError("no tile reaches sites.min_cover; lower it or enlarge sites.tile_m")
    else:
        fps = site_footprints(feats, sc.id_field, sc.point_size_m)
    snap = coarsest_res(cfg.bands)
    grids = {k: Grid.from_bounds(g.bounds, cfg.res, snap) for k, g in sorted(fps.items())}

    windows = []
    u = cfg.unlabeled
    if u.n > 0:
        if u.aoi:
            aoi = unary_union([to_crs(g, u.aoi_crs, crs) for g, _ in read_features(cfg.path(u.aoi))])
        else:   # bounding box of the input features (sites, or label polygons with from_labels)
            fb = unary_union([g for g, _ in feats]).bounds
            gb = union_grid(grids.values(), cfg.res, snap).bounds
            aoi = box(min(fb[0], gb[0]), min(fb[1], gb[1]), max(fb[2], gb[2]), max(fb[3], gb[3]))
        avoid = buffered_union([g.geometry for g in grids.values()], u.buffer_m)
        windows = sample_windows(aoi, u.n, u.win_px, cfg.res, snap, avoid, u.seed)

    region = union_grid([*grids.values(), *windows], cfg.res, snap)
    bbox = transform_bounds(crs, "EPSG:4326", *region.bounds, densify_pts=21)
    return Plan(crs, grids, fps, windows, region, bbox)


def create(cfg: Config, p: Plan, items_by_date: dict, path=None) -> str:
    """Create the HDF5 file and freeze dates and scene provenance into it."""
    path = str(path or cfg.path(cfg.output))
    preset = stac.get_preset(cfg.source)
    dates = list(items_by_date)
    if not dates:
        raise RuntimeError("the catalogue returned no scenes for this area and period")
    T, B = len(dates), len(cfg.bands)
    recs = {d: [it if "offset" in it else stac.provenance(it, preset, cfg.bands) for it in items_by_date[d]]
            for d in dates}   # STAC items, or provenance records copied from another cube
    with h5py.File(path, "w-") as f:
        f["dates"] = np.array(dates, dtype="S10")
        f["done"] = np.zeros(T, bool)
        f.create_dataset("scenes", data=[json.dumps(recs[d]) for d in dates], dtype=h5py.string_dtype())
        f["platform"] = np.array([recs[d][0]["platform"] for d in dates], dtype="S16")
        f["cloud_cover"] = np.array([np.nanmean([r["cloud_cover"] if r["cloud_cover"] is not None else np.nan
                                                 for r in recs[d]]) for d in dates], "f4")
        f["n_tiles"] = np.array([len(recs[d]) for d in dates], "i2")
        f.attrs.update(s2cube_version=__version__, name=cfg.name, bands=",".join(cfg.bands),
                       crs=p.crs.to_string(), res_m=cfg.res, source=cfg.source,
                       collection=preset.collection, value="reflectance x 10000 (harmonised); 0 = no data",
                       region=p.region.bounds, config=cfg.to_yaml(), config_dir=cfg.base_dir,
                       created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        sg = f.create_group("sites")
        for sid, g in p.sites.items():
            grp = sg.create_group(sid)
            grp.attrs.update(x0=g.x0, y1=g.y1, res_m=g.res, footprint_wkt=p.footprints[sid].wkt)
            grp.create_dataset("X", (T, B, *g.shape), np.uint16, chunks=(1, B, *g.shape),
                               compression="gzip", compression_opts=4, shuffle=True)
            grp.create_dataset("SCL", (T, *g.shape), np.uint8, chunks=(1, *g.shape), compression="gzip")
        ug = f.create_group("unlabeled")
        w = cfg.unlabeled.win_px
        N = len(p.windows)
        ug["origin"] = np.array([(g.x0, g.y1) for g in p.windows], "f8").reshape(N, 2)
        ug.attrs.update(win_px=w, buffer_m=cfg.unlabeled.buffer_m, seed=cfg.unlabeled.seed)
        ug.create_dataset("X", (N, T, B, w, w), np.uint16, chunks=(1, 1, B, w, w),
                          compression="gzip", compression_opts=4, shuffle=True)
        ug.create_dataset("SCL", (N, T, w, w), np.uint8, chunks=(1, 1, w, w), compression="gzip")
        q = f.create_group("qa")
        q.attrs["clear_scl"] = list(CLEAR_SCL)
        q.create_dataset("site_valid", (T, len(p.sites)), "f4", fillvalue=np.nan)
        q.create_dataset("site_clear", (T, len(p.sites)), "f4", fillvalue=np.nan)
        q.create_dataset("window_clear", (T, N), "f4", fillvalue=np.nan)
    return path


def read_groups(grids, res: float, snap: float, link_m: float) -> list:
    """Partition grids (sites and windows) into spatial groups read together on each date.

    Grids closer than ``link_m`` share one read; each group is read as the snapped bounding
    box of its members, so distant sites never force reading the empty land between them."""
    n = len(grids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    geoms = [g.geometry for g in grids]
    for a in range(n):
        for b in range(a + 1, n):
            if geoms[a].distance(geoms[b]) < link_m:
                parent[find(a)] = find(b)
    comp: dict = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    return [(union_grid([grids[i] for i in m], res, snap), m) for m in sorted(comp.values())]


def _read_date_groups(recs, groups, crs, bands, signer, asset_key, shift):
    return [(region, *read_date(recs, region, crs, bands, signer, asset_key, shift)) for region, _ in groups]


def _write_date(f, i, parts, groups, members):
    for (region, X, scl), (_, idx) in zip(parts, groups):
        for k in idx:
            kind, j, g = members[k]
            x, s = region.cut(X, g), region.cut(scl, g)
            if kind == "site":
                f[f"sites/{j}/X"][i] = x
                f[f"sites/{j}/SCL"][i] = s
                f["qa/site_valid"][i, k] = (s > 0).mean()     # sites come first in members
                f["qa/site_clear"][i, k] = np.isin(s, CLEAR_SCL).mean()
            else:
                f["unlabeled/X"][j, i] = x
                f["unlabeled/SCL"][j, i] = s
                f["qa/window_clear"][i, j] = np.isin(s, CLEAR_SCL).mean()


def fetch(path, limit: int = 0, workers: int | None = None, log=log_print, shifts=None,
          dates_idx=None) -> list[int]:
    """Download dates not yet ``done`` (or ``dates_idx`` with per-date ``shifts`` for
    co-registration). Returns the indices of the dates written in this call."""
    with Cube(path) as c:
        cfg, recs = c.config, c.scenes
        sites, windows, crs = [(s, c.site_grid(s)) for s in c.sites], c.windows, c.crs
        todo = dates_idx if dates_idx is not None else [i for i in range(c.n_dates) if not c.done[i]]
    todo = list(todo)[:limit or None]
    members = [("site", sid, g) for sid, g in sites] + [("window", k, g) for k, g in enumerate(windows)]
    groups = read_groups([m[2] for m in members], cfg.res, coarsest_res(cfg.bands), cfg.read_link_m)
    preset = stac.get_preset(cfg.source)
    signer = stac.make_signer(preset)
    shifts = shifts or {}
    px = sum(r.width * r.height for r, _ in groups)
    log(f"{len(todo)} date(s) to read, {len(groups)} read group(s), {px / 1e6:.1f} Mpx per date")
    t0, written = time.time(), []
    with h5py.File(path, "a") as f, ThreadPoolExecutor(workers or cfg.workers) as ex:
        dates = [d.decode() for d in f["dates"][:]]
        fut = {ex.submit(_read_date_groups, recs[i], groups, crs, cfg.bands, signer, preset.asset_key,
                         shifts.get(i, (0.0, 0.0))): i for i in todo}
        for n, fu in enumerate(as_completed(fut), 1):
            i = fut[fu]
            try:
                parts = fu.result()
            except Exception as e:  # keep going; the date stays not-done and is retried next run
                log(f"  {dates[i]}  FAILED: {type(e).__name__}: {str(e)[:200]}")
                continue
            _write_date(f, i, parts, groups, members)
            f["done"][i] = True
            f.flush()
            written.append(i)
            el = time.time() - t0
            with np.errstate(all="ignore"):
                cl = np.nanmean(f["qa/site_clear"][i]) if len(sites) else np.nan
            log(f"  [{n:>4}/{len(todo)}] {dates[i]}  clear (sites) {100 * cl:5.1f}%  "
                f"~{el / n * (len(todo) - n) / 60:.0f} min left")
    return sorted(written)


def build(cfg: Config, limit: int = 0, log=log_print) -> str:
    """Plan, search, create (if needed) and fetch. Safe to call again to resume."""
    path = cfg.path(cfg.output)
    if not path.exists():
        p = plan(cfg)
        log(f"CRS {p.crs.to_string()}, {len(p.sites)} site(s), {len(p.windows)} window(s)")
        if cfg.scenes_from:     # re-create a dataset on exactly the scenes recorded in another cube
            with Cube(cfg.path(cfg.scenes_from)) as old:
                by_date = {d: r for d, r in zip(old.dates, old.scenes) if cfg.start <= d <= cfg.end}
            missing = [b for b in cfg.bands if any(stac.get_preset(cfg.source).asset_key(b) not in rec["assets"]
                                                   for recs in by_date.values() for rec in recs)]
            if missing:
                raise ValueError(f"the scene records of {cfg.scenes_from} lack bands {missing}")
            log(f"{sum(map(len, by_date.values()))} scene(s) on {len(by_date)} date(s) from {cfg.scenes_from}")
        else:
            items = stac.search(p.bbox_lonlat, cfg.start, cfg.end, cfg.source, cfg.max_cloud)
            by_date = stac.group_by_date(items, p.crs.to_epsg())
            log(f"{len(items)} scene(s) on {len(by_date)} date(s)")
        create(cfg, p, by_date, path)
    else:
        log(f"resuming {path}")
    fetch(path, limit=limit, log=log)
    return str(path)


class Cube:
    """Read access to a cube file. Use as a context manager or call :meth:`close`."""

    def __init__(self, path, mode: str = "r"):
        self.path = str(path)
        self.f = h5py.File(self.path, mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.f.close()

    # -- metadata
    @property
    def config(self) -> Config:
        base = self.f.attrs.get("config_dir", str(Path(self.path).parent))
        return Config.from_dict(yaml.safe_load(self.f.attrs["config"]), base_dir=base)

    @property
    def dates(self) -> list[str]:
        return [d.decode() for d in self.f["dates"][:]]

    @property
    def n_dates(self) -> int:
        return len(self.f["dates"])

    @property
    def done(self) -> np.ndarray:
        return self.f["done"][:]

    @property
    def bands(self) -> list[str]:
        return self.f.attrs["bands"].split(",")

    @property
    def crs(self) -> CRS:
        return CRS.from_user_input(self.f.attrs["crs"])

    @property
    def res(self) -> float:
        return float(self.f.attrs["res_m"])

    @property
    def scenes(self) -> list[list[dict]]:
        return [json.loads(s) for s in self.f["scenes"].asstr()[:]]

    @property
    def sites(self) -> list[str]:
        return list(self.f["sites"])

    def site_grid(self, sid: str) -> Grid:
        g = self.f[f"sites/{sid}"]
        h, w = g["SCL"].shape[1:]
        return Grid(float(g.attrs["x0"]), float(g.attrs["y1"]), w, h, float(g.attrs["res_m"]))

    def site_footprint(self, sid: str):
        from shapely import wkt

        return wkt.loads(self.f[f"sites/{sid}"].attrs["footprint_wkt"])

    @property
    def windows(self) -> list[Grid]:
        w = int(self.f["unlabeled"].attrs["win_px"])
        return [Grid(float(x), float(y), w, w, self.res) for x, y in self.f["unlabeled/origin"][:]]

    @property
    def region(self) -> Grid:
        b = self.f.attrs["region"]
        return Grid.from_bounds(b, self.res)

    # -- data
    def site(self, sid: str, t=slice(None)) -> tuple[np.ndarray, np.ndarray]:
        """(X, SCL) of one site; ``t`` selects dates (index, slice or sorted index list)."""
        g = self.f[f"sites/{sid}"]
        return g["X"][t], g["SCL"][t]

    def window(self, k: int, t=slice(None)) -> tuple[np.ndarray, np.ndarray]:
        return self.f["unlabeled/X"][k, t], self.f["unlabeled/SCL"][k, t]

    def band_index(self, band: str) -> int:
        return self.bands.index(band)

    def clear_fraction(self) -> np.ndarray:
        """(T,) mean clear fraction over all sites per date (NaN where not downloaded)."""
        with np.errstate(all="ignore"):
            return np.nanmean(self.f["qa/site_clear"][:], axis=1) if self.f["qa/site_clear"].shape[1] else \
                np.full(self.n_dates, np.nan)


def reflectance(X: np.ndarray) -> np.ndarray:
    """uint16 cube values to float32 surface reflectance with NaN for no data."""
    r = X.astype("f4") / 10000.0
    r[X == 0] = np.nan
    return r


def site_boxes(cube: Cube) -> dict:
    return {s: box(*cube.site_grid(s).bounds) for s in cube.sites}
