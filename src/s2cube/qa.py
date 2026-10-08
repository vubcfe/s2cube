# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Quality reports and integrity manifest of a dataset."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .cube import Cube


def date_table(path) -> list[dict]:
    """One row per date: platform, catalogue cloud cover, tiles, observed valid/clear fraction
    over the sites, and co-registration status and shift when available."""
    with Cube(path) as c:
        f = c.f
        dates, done = c.dates, c.done
        valid = f["qa/site_valid"][:]
        clear = f["qa/site_clear"][:]
        has = "coreg" in f
        st = f["coreg/status"][:] if has else None
        meas = f["coreg/measured_m"][:] if has else None
        res = f["coreg/residual_m"][:] if has else None
        refined = f["qa/refined"][:] if "qa/refined" in f else None
        rows = []
        for i, d in enumerate(dates):
            with np.errstate(all="ignore"):
                v = float(np.nanmean(valid[i])) if valid.shape[1] else np.nan
                cl = float(np.nanmean(clear[i])) if clear.shape[1] else np.nan
            r = {"date": d, "done": int(done[i]), "platform": f["platform"][i].decode(),
                 "cloud_cover": round(float(f["cloud_cover"][i]), 2), "n_tiles": int(f["n_tiles"][i]),
                 "valid": round(v, 4), "clear": round(cl, 4)}
            if has:
                r.update(coreg_status=int(st[i]), east_m=round(float(meas[i, 0]), 2),
                         north_m=round(float(meas[i, 1]), 2), residual_m=round(float(np.hypot(*res[i])), 2))
            if refined is not None:
                r["esa_refined"] = int(refined[i])
            rows.append(r)
    return rows


def refinement_flag(item_id: str, preset, signer, session=None, max_bytes: int = 6_000_000) -> int:
    """ESA geometric refinement of a scene: 1 REFINED (against the Global Reference Image),
    0 NOT_REFINED, -1 unknown. Reads only the beginning of the datastrip metadata file."""
    import re

    import requests

    s = session or requests.Session()
    base = preset.search_url.rsplit("/search", 1)[0]
    r = s.get(f"{base}/collections/{preset.collection}/items/{item_id}", timeout=60)
    if not r.ok:
        return -1
    assets = r.json().get("assets", {})
    key = next((k for k in assets if "datastrip" in k.lower()), None)
    if key is None:
        return -1
    buf = b""
    with s.get(signer(assets[key]["href"]), timeout=120, stream=True) as rr:
        if not rr.ok:
            return -1
        for chunk in rr.iter_content(1 << 16):
            buf += chunk
            m = re.search(rb'Image_Refining\s+flag="(NOT_REFINED|REFINED)"', buf)
            if m:
                return int(m.group(1) == b"REFINED")
            if len(buf) > max_bytes:
                break
    return -1


def check_refinement(path, workers: int = 4, log=print) -> np.ndarray:
    """Store ESA's geometric refinement flag of every date in /qa/refined (1, 0, -1 unknown).

    A date counts as refined only if all its scenes are; dates whose refinement failed are the
    ones expected to show offsets of about one pixel or more."""
    from concurrent.futures import ThreadPoolExecutor

    import h5py

    from . import stac

    with Cube(path) as c:
        preset = stac.get_preset(c.f.attrs["source"])
        scenes = c.scenes
    signer = stac.make_signer(preset)
    ids = sorted({r["id"] for recs in scenes for r in recs})
    with ThreadPoolExecutor(workers) as ex:
        flag = dict(zip(ids, ex.map(lambda i: refinement_flag(i, preset, signer), ids)))
    out = np.array([min(flag[r["id"]] for r in recs) if all(flag[r["id"]] >= 0 for r in recs) else -1
                    for recs in scenes], "i1")
    with h5py.File(path, "a") as f:
        if "qa/refined" in f:
            del f["qa/refined"]
        f["qa/refined"] = out
        f["qa/refined"].attrs["meaning"] = "1 REFINED, 0 NOT_REFINED (ESA Image_Refining flag), -1 unknown"
    log(f"refined {int((out == 1).sum())}, not refined {int((out == 0).sum())}, unknown {int((out == -1).sum())}")
    return out


def summary(path) -> dict:
    """Headline numbers of a cube (used by ``s2cube info``)."""
    rows = date_table(path)
    with Cube(path) as c:
        out = {"file": str(path), "crs": c.crs.to_string(), "bands": c.bands, "res_m": c.res,
               "n_dates": len(rows), "n_done": sum(r["done"] for r in rows),
               "period": [rows[0]["date"], rows[-1]["date"]] if rows else None,
               "n_sites": len(c.sites), "n_windows": len(c.windows),
               "dates_per_year": {}, "platform": {}}
    for r in rows:
        out["dates_per_year"][r["date"][:4]] = out["dates_per_year"].get(r["date"][:4], 0) + 1
        out["platform"][r["platform"]] = out["platform"].get(r["platform"], 0) + 1
    cl = np.array([r["clear"] for r in rows if r["done"]], float)
    if cl.size:
        out["clear_dates"] = {">=0.5": int((cl >= 0.5).sum()), "<0.05": int((cl < 0.05).sum())}
    if rows and "esa_refined" in rows[0]:
        ref = [r["esa_refined"] for r in rows]
        out["esa_refinement"] = {"refined": ref.count(1), "not_refined": ref.count(0), "unknown": ref.count(-1)}
    if rows and "coreg_status" in rows[0]:
        st = [r["coreg_status"] for r in rows]
        resid = np.array([r["residual_m"] for r in rows if r["coreg_status"] == 2], float)
        resid = resid[~np.isnan(resid)]
        out["coreg"] = {"unreliable": st.count(0), "within_threshold": st.count(1), "corrected": st.count(2)}
        if resid.size:
            out["coreg"]["residual_m"] = {"median": round(float(np.median(resid)), 2),
                                          "max": round(float(resid.max()), 2)}
    return out


def write_table(rows, path):
    with open(path, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(files, out) -> str:
    """``sha256sum``-compatible manifest of ``files`` (paths relative to the manifest)."""
    out = Path(out)
    lines = [f"{sha256(p)}  {Path(p).resolve().relative_to(out.resolve().parent)}" for p in files]
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def check_manifest(manifest) -> list[str]:
    """Names of files whose checksum differs from the manifest (empty list = all good)."""
    m = Path(manifest)
    bad = []
    for line in m.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            p = m.parent / name.strip()
            if not p.exists() or sha256(p) != digest:
                bad.append(name.strip())
    return bad


def dumps(d) -> str:
    return json.dumps(d, indent=1, ensure_ascii=False)
