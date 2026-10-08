# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Minimal vector input: GeoJSON / newline-delimited GeoJSON natively, other formats via pyogrio
or fiona when one of them is installed."""
from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import box, shape

from .geo import to_crs


def read_features(path) -> list[tuple[object, dict]]:
    """[(shapely geometry, properties)] from a vector file. Invalid polygons are repaired."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        data = json.loads(path.read_text())
        feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    elif suffix in (".geojsonl", ".geojsons", ".ndjson", ".jsonl"):
        feats = []
        for line in path.read_text().splitlines():
            line = line.strip().strip("\x1e").rstrip(",")
            if line.startswith("{"):
                feats.append(json.loads(line))
    else:
        return _read_ogr(path)
    out = []
    for f in feats:
        if not f.get("geometry"):
            continue
        g = shape(f["geometry"])
        out.append((g if g.is_valid else g.buffer(0), dict(f.get("properties") or {})))
    return out


def _read_ogr(path):
    try:
        import pyogrio

        gdf = pyogrio.read_dataframe(path)
        return [(g, {k: v for k, v in row.items() if k != "geometry"})
                for g, (_, row) in zip(gdf.geometry, gdf.iterrows())]
    except ImportError:
        pass
    try:
        import fiona

        with fiona.open(path) as src:
            return [(shape(f["geometry"]), dict(f["properties"])) for f in src]
    except ImportError:
        raise ImportError(f"reading {path.suffix} files needs pyogrio or fiona; "
                          "or convert the file to GeoJSON") from None


def read_geometries(path, src_crs, dst_crs) -> list[tuple[object, dict]]:
    """Features of ``path`` reprojected from ``src_crs`` to ``dst_crs``."""
    return [(to_crs(g, src_crs, dst_crs), p) for g, p in read_features(path)]


def site_footprints(features, id_field: str, point_size_m: float) -> dict[str, object]:
    """{site id: footprint polygon} in projected metres. Points become squares, polygons
    are kept; ids must be unique."""
    out = {}
    for i, (g, p) in enumerate(features):
        sid = str(p.get(id_field, f"s{i + 1:03d}"))
        if sid in out:
            raise ValueError(f"duplicate site id {sid!r}")
        if g.geom_type == "Point":
            h = point_size_m / 2
            g = box(g.x - h, g.y - h, g.x + h, g.y + h)
        out[sid] = g
    return out
