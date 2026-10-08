# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Measure and correct per-date geolocation offsets of a cube.

For every date and every sufficiently clear location (labelled site or unlabelled window),
the visible-band mean image is resampled (cubic spline) onto a fine grid and its gradient magnitude is
matched against the gradient of a reference image by normalised cross-correlation over all
integer shifts within ``max_shift_m``, refined to sub-pixel precision by a parabolic fit.
Shifts of individual locations are combined into one shift per date by the median; a date
is *reliable* when at least ``min_sites`` locations agree within ``max_spread_m``.

Dates with a reliable offset above ``threshold_m`` are corrected by re-reading the original
scenes on a shifted grid (bilinear for bands, nearest for SCL): the stored data are never
resampled twice. Where two MGRS tiles meet, each tile's valid area is eroded by one native
pixel before mosaicking so that interpolation never uses no-data; the neighbouring tile fills
the gap. The result is verified by measuring again.

References:

* ``self``     temporal median of the clearest dates of the cube itself; aligns all dates
               to their common geometry (what matters for time-series learning);
* ``raster``   any georeferenced image (e.g. an orthophoto), read with GDAL;
* ``xyz``      a web-map tile service ``{z}/{x}/{y}``; check the provider's terms of use.
"""
from __future__ import annotations

import csv
import io
import math
import shutil
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image
from rasterio.enums import Resampling
from scipy import ndimage
from scipy.signal import fftconvolve

from ._log import log_print
from .config import CoregConfig
from .cube import Cube, fetch
from .geo import Grid, points_to_lonlat
from .reader import CLEAR_SCL

STATUS = {0: "not measured / unreliable", 1: "measured, within threshold", 2: "corrected"}


# ---------------------------------------------------------------- matching
def gradient(a: np.ndarray) -> np.ndarray:
    return np.hypot(*np.gradient(a))


def _norm(a):
    a = a - a.mean()
    return a / (a.std() + 1e-9)


def ncc_surface(S: np.ndarray, E: np.ndarray, m: int) -> np.ndarray:
    """Normalised cross-correlation of the centre of ``S`` (cropped by ``m``) with ``E`` for all
    integer shifts within +-m; element [dy + m, dx + m] compares S[r, c] with E[r + dy, c + dx].

    Computed with one FFT correlation and integral images for the local mean and standard
    deviation of ``E``; identical to the direct definition up to rounding.
    """
    T = S[m:-m, m:-m].astype("f8")
    T = (T - T.mean()) / (T.std() + 1e-9)
    n = T.size
    E = E.astype("f8")
    num = fftconvolve(E, T[::-1, ::-1], mode="valid")          # sum(T * E_window) per shift
    th, tw = T.shape

    def box_sum(a):
        c = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        c[1:, 1:] = a.cumsum(0).cumsum(1)
        return c[th:, tw:] - c[:-th, tw:] - c[th:, :-tw] + c[:-th, :-tw]

    mean = box_sum(E) / n
    var = np.maximum(box_sum(E * E) / n - mean ** 2, 0)
    return num / (n * (np.sqrt(var) + 1e-9))


def ncc_shift(S: np.ndarray, E: np.ndarray, m: int) -> tuple[float, float, float]:
    """Sub-pixel (dx, dy) such that E[r + dy, c + dx] best matches S[r, c], searched within
    +-m pixels, and the peak normalised cross-correlation."""
    grid = ncc_surface(S, E, m)
    iy, ix = np.unravel_index(np.argmax(grid), grid.shape)

    def parabola(v, i):
        if 0 < i < len(v) - 1:
            d = v[i - 1] - 2 * v[i] + v[i + 1]
            return i + (0.5 * (v[i - 1] - v[i + 1]) / d if d else 0.0)
        return float(i)

    return parabola(grid[iy], ix) - m, parabola(grid[:, ix], iy) - m, float(grid.max())


def fine_coords(g: Grid, fine: float):
    """Projected coordinates of the fine matching grid of ``g`` and their (row, col) position
    in ``g``'s pixel space."""
    nx, ny = int(round(g.width * g.res / fine)), int(round(g.height * g.res / fine))
    xs = g.x0 + (np.arange(nx) + 0.5) * fine
    ys = g.y1 - (np.arange(ny) + 0.5) * fine
    rr, cc = np.meshgrid((g.y1 - ys) / g.res - 0.5, (xs - g.x0) / g.res - 0.5, indexing="ij")
    return xs, ys, rr, cc


def to_fine(a: np.ndarray, rr, cc) -> np.ndarray:
    """Cubic-spline resampling onto the fine grid. Bilinear resampling biases sub-pixel shifts
    towards whole pixels (errors up to ~0.15 pixel on aliased 10 m texture); cubic removes
    most of this bias."""
    return ndimage.map_coordinates(a, [rr, cc], order=3, mode="nearest")


def brightness(X: np.ndarray, idx) -> np.ndarray:
    return X[idx].astype("f4").mean(0)


# ---------------------------------------------------------------- references
class SelfReference:
    """Temporal median of the visible-band brightness over the ``n`` clearest dates.

    ``shifts`` ({date index: (east, north) m}) from a previous pass are applied to the
    reference dates before taking the median, so that misregistered dates do not blur or
    bias the composite (see :func:`measure`, which iterates this)."""

    iterative = True

    def __init__(self, n: int = 15, min_clear: float = 0.9, allowed=None):
        self.n, self.min_clear = n, min_clear
        self.allowed = None if allowed is None else set(int(t) for t in allowed)

    def dates(self, clear) -> list[int]:
        order = np.argsort(-np.nan_to_num(clear, nan=-1), kind="stable")
        return [int(t) for t in order if clear[t] >= self.min_clear
                and (self.allowed is None or int(t) in self.allowed)][:self.n]

    def __call__(self, loc, g: Grid, fine: float, shifts=None) -> np.ndarray:
        order = self.dates(loc["clear"])
        if len(order) < 3:
            return None
        stack = []
        for t in order:
            a = brightness(loc["X"](t), loc["rgb"])
            e, n = (shifts or {}).get(t, (0.0, 0.0))
            if (e, n) != (0.0, 0.0):        # move content by (e, n) metres
                a = ndimage.shift(a, (-n / g.res, e / g.res), order=3, mode="nearest")
            stack.append(a)
        _, _, rr, cc = fine_coords(g, fine)
        return ndimage.gaussian_filter(to_fine(np.median(np.stack(stack), 0), rr, cc), 1)


class RasterReference:
    """Any raster readable by GDAL; bands are averaged to one brightness channel."""

    iterative = False

    def __init__(self, path, crs):
        self.path, self.crs = str(path), crs

    def __call__(self, loc, g: Grid, fine: float, shifts=None) -> np.ndarray:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.vrt import WarpedVRT

        nx, ny = int(round(g.width * g.res / fine)), int(round(g.height * g.res / fine))
        with rasterio.open(self.path) as src, WarpedVRT(
                src, crs=self.crs, transform=Affine(fine, 0, g.x0, 0, -fine, g.y1), width=nx, height=ny,
                resampling=Resampling.bilinear) as vrt:
            a = vrt.read().astype("f4").mean(0)
        return ndimage.gaussian_filter(a, g.res / fine / 2.355 * 2)


class XYZReference:
    """Web-map tiles (Web Mercator, 256 px) sampled on the fine grid and blurred to ~10 m."""

    iterative = False

    def __init__(self, url: str, zoom: int, crs, cache_dir):
        import hashlib

        if not all(k in url for k in ("{z}", "{x}", "{y}")):
            raise ValueError("coreg.xyz_url must contain {z}, {x} and {y}")
        self.url, self.zoom, self.crs = url, zoom, crs
        self.cache = Path(cache_dir) / hashlib.sha1(url.encode()).hexdigest()[:12]   # one cache per provider
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "s2cube (https://github.com/vubcfe/s2cube)"

    def tile(self, x: int, y: int) -> np.ndarray:
        f = self.cache / f"{self.zoom}_{x}_{y}.png"
        if not f.exists():
            r = self.session.get(self.url.format(z=self.zoom, x=x, y=y), timeout=60)
            r.raise_for_status()
            Image.open(io.BytesIO(r.content)).convert("RGB").save(f)
        return np.asarray(Image.open(f).convert("RGB"), dtype="f4").mean(2)

    def __call__(self, loc, g: Grid, fine: float, shifts=None) -> np.ndarray:
        xs, ys, _, _ = fine_coords(g, fine)
        X, Y = np.meshgrid(xs, ys)
        lon, lat = points_to_lonlat(X, Y, self.crs)
        n = 2 ** self.zoom
        px = (lon + 180) / 360 * n * 256
        lr = np.radians(lat)
        py = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * n * 256
        tx0, ty0 = int(px.min() // 256), int(py.min() // 256)
        tx1, ty1 = int(px.max() // 256), int(py.max() // 256)
        big = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), "f4")
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                big[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = self.tile(tx, ty)
        a = ndimage.map_coordinates(big, [py - ty0 * 256 - 0.5, px - tx0 * 256 - 0.5], order=1)
        return ndimage.gaussian_filter(a, g.res / fine / 2.355 * 2)


def make_reference(cfg: CoregConfig, crs, base: Path, refined=None):
    if cfg.reference == "self":
        allowed = None
        if cfg.self_refined_only:
            if refined is None:
                raise ValueError("coreg.self_refined_only needs ESA refinement flags: run `s2cube info --refinement`")
            allowed = np.flatnonzero(np.asarray(refined) == 1)
        return SelfReference(cfg.self_dates, cfg.min_clear, allowed)
    if cfg.reference == "raster":
        if not cfg.reference_path:
            raise ValueError("coreg.reference_path is required for reference 'raster'")
        p = Path(cfg.reference_path)
        return RasterReference(p if p.is_absolute() else base / p, crs)
    if not cfg.xyz_url:
        raise ValueError("coreg.xyz_url is required for reference 'xyz'")
    return XYZReference(cfg.xyz_url, cfg.xyz_zoom, crs, base / "xyz_cache")


# ---------------------------------------------------------------- measure / apply
def _locations(c: Cube):
    missing = [b for b in ("B02", "B03", "B04") if b not in c.bands]
    if missing:
        raise ValueError(f"co-registration matches visible brightness and needs bands {missing} in the cube")
    rgb = [c.band_index(b) for b in ("B04", "B03", "B02")]
    locs = []
    clear_s = c.f["qa/site_clear"][:]
    for j, s in enumerate(c.sites):
        g = c.f[f"sites/{s}"]
        locs.append({"name": s, "grid": c.site_grid(s), "clear": clear_s[:, j], "rgb": rgb,
                     "X": lambda t, g=g: g["X"][t], "SCL": lambda t, g=g: g["SCL"][t]})
    clear_w = c.f["qa/window_clear"][:]
    for k, wg in enumerate(c.windows):
        locs.append({"name": f"w{k:04d}", "grid": wg, "clear": clear_w[:, k], "rgb": rgb,
                     "X": lambda t, k=k: c.f["unlabeled/X"][k, t],
                     "SCL": lambda t, k=k: c.f["unlabeled/SCL"][k, t]})
    return locs


def _measure_dates(prepared, idx, dates, cfg: CoregConfig, m: int) -> list[dict]:
    rows = []
    for i in idx:
        shifts = []
        for loc, gE, rr, cc in prepared:
            if not loc["clear"][i] >= cfg.min_clear:
                continue
            S = to_fine(brightness(loc["X"](i), loc["rgb"]), rr, cc)
            dx, dy, ncc = ncc_shift(gradient(ndimage.gaussian_filter(S, 1)), gE, m)
            if ncc >= cfg.min_ncc:
                shifts.append((dx * cfg.fine_res, -dy * cfg.fine_res, ncc))
        row = {"date": dates[i], "n_sites": len(shifts), "east_m": np.nan, "north_m": np.nan,
               "spread_m": np.nan, "ncc": np.nan, "reliable": 0}
        if shifts:
            s = np.array(shifts)
            e, n = float(np.median(s[:, 0])), float(np.median(s[:, 1]))
            spread = float(np.median(np.hypot(s[:, 0] - e, s[:, 1] - n)))
            row.update(east_m=round(e, 2), north_m=round(n, 2), spread_m=round(spread, 2),
                       ncc=round(float(np.median(s[:, 2])), 3),
                       reliable=int(len(shifts) >= cfg.min_sites and spread <= cfg.max_spread_m))
        rows.append(row)
    return rows


def measure(path, cfg: CoregConfig | None = None, dates_idx=None, out_csv=None, log=log_print) -> list[dict]:
    """Measure the offset of every downloaded date (or ``dates_idx``). Returns one row per date:
    date, n_sites, east_m, north_m, spread_m, ncc, reliable. ``east/north > 0`` means the
    image must move east/north by that many metres to match the reference.

    With the ``self`` reference the measurement is iterated (``cfg.self_iterations``): the
    reference dates are aligned with the offsets of the previous pass and the composite is
    rebuilt, until the offsets change by less than 0.1 m.
    """
    with Cube(path) as c:
        cfg = cfg or c.config.coreg
        ref = make_reference(cfg, c.crs, Path(c.path).parent,
                             c.f["qa/refined"][:] if "qa/refined" in c.f else None)
        dates, done = c.dates, c.done
        idx = [i for i in (dates_idx if dates_idx is not None else range(len(dates))) if done[i]]
        m = int(round(cfg.max_shift_m / cfg.fine_res))
        locs = _locations(c)
        n_iter = cfg.self_iterations if getattr(ref, "iterative", False) else 1
        ref_shifts: dict = {}
        rows: list[dict] = []
        for it in range(n_iter):
            prepared = []
            for loc in locs:
                E = ref(loc, loc["grid"], cfg.fine_res, ref_shifts)
                if E is None or min(E.shape) <= 2 * m + 4:
                    continue
                _, _, rr, cc = fine_coords(loc["grid"], cfg.fine_res)
                prepared.append((loc, gradient(E), rr, cc))
            if it == 0:
                log(f"{len(prepared)} location(s) with a usable reference, {len(idx)} date(s)")
            if n_iter > 1:
                # the reference dates of every location must be measured too, even if not requested
                ref_dates = sorted({t for loc, *_ in prepared for t in ref.dates(loc["clear"])} | set(idx))
                all_rows = _measure_dates(prepared, [t for t in ref_dates if done[t]], dates, cfg, m)
                pos = {d: i for i, d in enumerate(dates)}
                meas = {pos[r["date"]]: (r["east_m"], r["north_m"]) for r in all_rows if r["reliable"]}
                # anchor the frame at the median geometry of the reference dates
                anchor = [meas[t] for loc, *_ in prepared for t in ref.dates(loc["clear"]) if t in meas]
                e0, n0 = (np.median([a[0] for a in anchor]), np.median([a[1] for a in anchor])) if anchor else (0, 0)
                new = {t: (e - e0, n - n0) for t, (e, n) in meas.items()}
                change = max((np.hypot(new[t][0] - ref_shifts.get(t, (0, 0))[0],
                                       new[t][1] - ref_shifts.get(t, (0, 0))[1]) for t in new), default=0.0)
                keep = set(idx)
                rows = [r for r in all_rows if pos[r["date"]] in keep]
                rows = [{**r, "east_m": round(float(new[pos[r["date"]]][0]), 2),
                         "north_m": round(float(new[pos[r["date"]]][1]), 2)} if pos[r["date"]] in new else r
                        for r in rows]
                log(f"  pass {it + 1}: largest change of a reference offset {change:.2f} m")
                ref_shifts = new
                if change < 0.1:
                    break
            else:
                rows = _measure_dates(prepared, idx, dates, cfg, m)
        for row in rows:
            log(f"  {row['date']}  n={row['n_sites']:3d}  E {row['east_m']:+6.2f}  N {row['north_m']:+6.2f} m"
                f"  spread {row['spread_m']:5.2f}" + ("" if row["reliable"] else "  (unreliable)"))
    if out_csv:
        with open(out_csv, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]) if rows else ["date"])
            w.writeheader()
            w.writerows(rows)
    return rows


def _ensure_group(f, T):
    g = f.require_group("coreg")
    for k, shape, dt, fill in (("status", (T,), "i1", 0), ("shift_m", (T, 2), "f4", 0.0),
                               ("measured_m", (T, 2), "f4", np.nan), ("residual_m", (T, 2), "f4", np.nan)):
        if k not in g:
            g.create_dataset(k, shape, dt, fillvalue=fill)
    return g


def record(path, rows, cfg: CoregConfig, residual: bool = False):
    """Store measurement rows in /coreg (``residual=True`` for a verification pass)."""
    with h5py.File(path, "a") as f:
        dates = [d.decode() for d in f["dates"][:]]
        g = _ensure_group(f, len(dates))
        g.attrs.update(reference=cfg.reference, threshold_m=cfg.threshold_m, min_sites=cfg.min_sites,
                       max_spread_m=cfg.max_spread_m, status_codes=str(STATUS))
        pos = {d: i for i, d in enumerate(dates)}
        for r in rows:
            i = pos[r["date"]]
            if not r["reliable"]:
                continue
            if residual:
                g["residual_m"][i] = (r["east_m"], r["north_m"])
            elif g["status"][i] != 2:
                g["measured_m"][i] = (r["east_m"], r["north_m"])
                g["status"][i] = 1 if math.hypot(r["east_m"], r["north_m"]) <= cfg.threshold_m else 0


def apply(path, cfg: CoregConfig | None = None, rows=None, backup: bool = True, log=log_print) -> list[int]:
    """Correct every date whose reliable offset exceeds the threshold. Returns their indices."""
    with Cube(path) as c:
        cfg = cfg or c.config.coreg
    rows = rows if rows is not None else measure(path, cfg, log=log)
    record(path, rows, cfg)
    with h5py.File(path, "r") as f:
        dates = [d.decode() for d in f["dates"][:]]
        status = f["coreg/status"][:]
    pos = {d: i for i, d in enumerate(dates)}
    todo = {pos[r["date"]]: (r["east_m"], r["north_m"]) for r in rows
            if r["reliable"] and status[pos[r["date"]]] != 2
            and math.hypot(r["east_m"], r["north_m"]) > cfg.threshold_m}
    if not todo:
        log("no date needs correction")
        return []
    if backup:
        b = Path(path).with_suffix(".orig.h5")
        if not b.exists():
            log(f"backup -> {b.name}")
            shutil.copy2(path, b)
    written = fetch(path, shifts=todo, dates_idx=sorted(todo), log=log)
    with h5py.File(path, "a") as f:
        for i in written:
            f["coreg/shift_m"][i] = todo[i]
            f["coreg/status"][i] = 2
    if len(written) < len(todo):
        log(f"{len(todo) - len(written)} date(s) failed and keep their original geometry; run again")
    return written


def verify(path, cfg: CoregConfig | None = None, out_csv=None, log=log_print) -> list[dict]:
    """Measure corrected dates again and store the residual offsets."""
    with h5py.File(path, "r") as f:
        idx = list(np.flatnonzero(f["coreg/status"][:] == 2))
    with Cube(path) as c:
        cfg = cfg or c.config.coreg
    rows = measure(path, cfg, dates_idx=idx, out_csv=out_csv, log=log)
    record(path, rows, cfg, residual=True)
    return rows


__all__ = ["measure", "apply", "verify", "record", "ncc_shift", "SelfReference", "RasterReference",
           "XYZReference", "CLEAR_SCL"]
