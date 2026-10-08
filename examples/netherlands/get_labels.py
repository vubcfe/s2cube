# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Download Dutch crop parcels (BRP Gewaspercelen, public domain) for a bounding box from the
PDOK OGC API Features service and write them as GeoJSON (lon/lat).

    python get_labels.py                       # default box in Flevoland, latest year
    python get_labels.py --bbox 5.55 52.48 5.75 52.58 --out brp.geojson

Source: Rijksdienst voor Ondernemend Nederland (RVO), https://api.pdok.nl/rvo/gewaspercelen/ogc/v1
"""
import argparse
import json
import time

import requests

API = "https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas/items"


def fetch(bbox, page=1000):
    url = f"{API}?f=json&limit={page}&bbox={','.join(map(str, bbox))}"
    feats = []
    while url:
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(3 * 2 ** attempt)
        j = r.json()
        feats += j["features"]
        url = next((lk["href"] for lk in j.get("links", []) if lk.get("rel") == "next"), None)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, default=[5.55, 52.48, 5.75, 52.58],
                    metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    ap.add_argument("--out", default="brp.geojson")
    a = ap.parse_args()
    feats = fetch(a.bbox)
    keep = [{"type": "Feature", "geometry": f["geometry"],
             "properties": {k: f["properties"].get(k) for k in ("gewas", "gewascode", "category", "jaar")}}
            for f in feats if f.get("geometry")]
    with open(a.out, "w") as fo:
        json.dump({"type": "FeatureCollection", "features": keep}, fo)
    print(f"{len(keep)} parcels -> {a.out}")


if __name__ == "__main__":
    main()
