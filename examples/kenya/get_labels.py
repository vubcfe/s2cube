# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Download the PlantVillage Kenya ground-reference crop-type polygons (2019, western Kenya).

    python get_labels.py            # -> kenya_fields.geojson

Dataset: PlantVillage (2019), "PlantVillage Kenya Ground Reference Crop Type Dataset",
Radiant MLHub, https://doi.org/10.34911/rdnt.u41j87, licence CC BY-SA 4.0. The file is
downloaded at run time and not redistributed with s2cube.
"""
import argparse

import requests

URL = ("https://data.source.coop/radiantearth/african-crops-kenya-01/"
       "ref_african_crops_kenya_01_tile_001.geojson")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kenya_fields.geojson")
    a = ap.parse_args()
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    with open(a.out, "wb") as fo:
        fo.write(r.content)
    print(f"{len(r.json()['features'])} fields -> {a.out}")


if __name__ == "__main__":
    main()
