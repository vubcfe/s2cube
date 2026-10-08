# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Search public STAC catalogues for Sentinel-2 L2A scenes and group them per acquisition date.

Two catalogues are supported out of the box (``PRESETS``): Microsoft Planetary Computer and
Element 84 Earth Search on AWS. Both serve the same ESA L2A product as cloud-optimised
GeoTIFFs but use different asset names; a preset maps the ESA band names used throughout
s2cube (``B02`` ... ``B12``, ``SCL``) to the catalogue's asset keys.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests

ESA_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
NATIVE_RES = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20, "B08": 10,
              "B8A": 20, "B09": 60, "B11": 20, "B12": 20, "SCL": 20}


@dataclass(frozen=True)
class Preset:
    """How to query one STAC API and find each band among an item's assets."""

    name: str
    search_url: str
    collection: str
    assets: dict = field(default_factory=dict)
    signer: str | None = None

    def asset_key(self, band: str) -> str:
        return self.assets.get(band, band)


PRESETS = {
    "planetary-computer": Preset(
        "planetary-computer",
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        "sentinel-2-l2a",
        {},
        signer="planetary-computer",
    ),
    "earth-search": Preset(
        "earth-search",
        "https://earth-search.aws.element84.com/v1/search",
        "sentinel-2-l2a",
        {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
         "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
         "B11": "swir16", "B12": "swir22", "SCL": "scl"},
    ),
}


def get_preset(name_or_preset) -> Preset:
    if isinstance(name_or_preset, Preset):
        return name_or_preset
    try:
        return PRESETS[name_or_preset]
    except KeyError:
        raise ValueError(f"unknown source preset {name_or_preset!r}; choose from {sorted(PRESETS)}") from None


# ---------------------------------------------------------------- item properties
def item_date(item) -> str:
    return item["properties"]["datetime"][:10]


def item_epsg(item) -> int | None:
    p = item["properties"]
    if p.get("proj:epsg"):
        return int(p["proj:epsg"])
    code = p.get("proj:code")
    if code and str(code).upper().startswith("EPSG:"):
        return int(str(code).split(":")[1])
    return None


def item_tile(item) -> str:
    p = item["properties"]
    if "s2:mgrs_tile" in p:
        return p["s2:mgrs_tile"]
    if "grid:code" in p:
        return p["grid:code"].replace("MGRS-", "")
    return item["id"]


def item_baseline(item) -> float:
    try:
        return float(item["properties"].get("s2:processing_baseline", 0))
    except (TypeError, ValueError):
        return 0.0


def item_offset(item, preset: Preset | None = None) -> int:
    """DN offset to subtract so that reflectance = DN / 10000 for every processing baseline.

    ESA added a +1000 offset from processing baseline 04.00 (25 Jan 2022). Earth Search
    removes it already when converting to COG and says so in
    ``earthsearch:boa_offset_applied`` (its ``raster:bands`` metadata still lists an offset of
    -0.1, which must therefore not be used). Planetary Computer serves the original DN.
    """
    p = item["properties"]
    if p.get("earthsearch:boa_offset_applied") is True:
        return 0
    return 1000 if item_baseline(item) >= 4.0 else 0


def item_generation(item) -> str:
    p = item["properties"]
    return p.get("s2:generation_time") or p.get("updated") or p.get("created") or ""


# ---------------------------------------------------------------- search
def search(bbox_lonlat, start: str, end: str, preset="planetary-computer", max_cloud: float | None = None,
           session: requests.Session | None = None, page_size: int = 500, retries: int = 5) -> list[dict]:
    """All L2A items intersecting ``bbox_lonlat`` = (west, south, east, north) between two dates."""
    preset = get_preset(preset)
    s = session or requests.Session()
    body = {"collections": [preset.collection], "bbox": list(bbox_lonlat),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": page_size}
    if max_cloud is not None:
        body["query"] = {"eo:cloud_cover": {"lte": max_cloud}}
    url, method, items = preset.search_url, "POST", []
    while url:
        for attempt in range(retries):
            try:
                r = s.post(url, json=body, timeout=120) if method == "POST" else s.get(url, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * 2 ** attempt)
        page = r.json()
        items += page.get("features", [])
        nxt = [lk for lk in page.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            break
        url, method = nxt[0]["href"], nxt[0].get("method", "GET").upper()
        if method == "POST":
            body = {**body, **nxt[0].get("body", {})} if nxt[0].get("merge") else nxt[0].get("body", body)
    return items


def group_by_date(items, target_epsg: int | None = None) -> dict[str, list[dict]]:
    """{date: [items]} keeping, per (date, MGRS tile), only the most recently processed item.

    Within a date, tiles already in ``target_epsg`` come first so that they take priority when
    the tiles are mosaicked; tiles from a neighbouring UTM zone are reprojected on read.
    """
    best = {}
    for it in items:
        k = (item_date(it), item_tile(it))
        if k not in best or item_generation(it) > item_generation(best[k]):
            best[k] = it
    out: dict[str, list[dict]] = {}
    for (d, _), it in best.items():
        out.setdefault(d, []).append(it)
    for d in out:
        out[d].sort(key=lambda it: (item_epsg(it) != target_epsg, item_tile(it)))
    return dict(sorted(out.items()))


def provenance(item, preset: Preset, bands) -> dict:
    """Compact, JSON-serialisable record of an item: enough to read it again without searching."""
    keys = [preset.asset_key(b) for b in (*bands, "SCL")]
    return {"id": item["id"], "tile": item_tile(item), "epsg": item_epsg(item),
            "platform": item["properties"].get("platform", ""),
            "cloud_cover": item["properties"].get("eo:cloud_cover"),
            "baseline": item_baseline(item), "offset": item_offset(item, preset),
            "generation_time": item_generation(item),
            "assets": {k: item["assets"][k]["href"] for k in keys}}


# ---------------------------------------------------------------- signing
class PlanetaryComputerSigner:
    """Appends a short-lived SAS token to Planetary Computer blob URLs (thread-safe, cached)."""

    TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"

    def __init__(self, collection: str = "sentinel-2-l2a", session: requests.Session | None = None):
        self.collection, self.session = collection, session or requests.Session()
        self._lock, self._token, self._expiry = threading.Lock(), None, 0.0

    def token(self, retries: int = 6) -> str:
        with self._lock:
            if time.time() > self._expiry - 600:
                for attempt in range(retries):
                    try:
                        r = self.session.get(self.TOKEN_URL.format(collection=self.collection), timeout=60)
                        r.raise_for_status()
                        j = r.json()
                        break
                    except (requests.RequestException, ValueError):   # incl. throttling (429)
                        if attempt == retries - 1:
                            raise
                        time.sleep(5 * 2 ** attempt)
                self._token = j["token"]
                self._expiry = datetime.fromisoformat(j["msft:expiry"].replace("Z", "+00:00")).timestamp()
            return self._token

    def __call__(self, href: str) -> str:
        if href.startswith("/") or "blob.core.windows.net" not in href:
            return href
        return f"{href}{'&' if '?' in href else '?'}{self.token()}"


def make_signer(preset: Preset):
    if preset.signer == "planetary-computer":
        return PlanetaryComputerSigner(preset.collection)
    return lambda href: href
