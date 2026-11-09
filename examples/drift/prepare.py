# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Write grass2024.geojson: Flevoland grassland parcels of 2024 and whether they changed by 2025.

    python prepare.py ../netherlands/brp.geojson
"""
import argparse
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import box

URL = "https://data.source.coop/ftw/harmonized-field-data/nl/year=2024/nl-2024.parquet"
BBOX = (5.55, 52.48, 5.75, 52.58)
RD = 28992


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brp2025", help="brp.geojson written by examples/netherlands/get_labels.py")
    ap.add_argument("--min-area", type=float, default=5000, help="smallest parcel in square metres")
    a = ap.parse_args()
    path = Path("nl-2024.parquet")
    if not path.exists():
        with requests.get(URL, stream=True, timeout=120) as r, open(path, "wb") as fo:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                fo.write(chunk)
    parcels = gpd.read_parquet(path, columns=["geometry", "crop:name", "metrics:area"])
    west, south, east, north = gpd.GeoSeries([box(*BBOX)], crs=4326).to_crs(parcels.crs).total_bounds
    parcels = parcels.cx[west:east, south:north].to_crs(RD)
    grass = parcels[parcels["crop:name"].str.lower().str.contains("grasland")
                    & (parcels["metrics:area"] > a.min_area)].reset_index(drop=True)
    later = gpd.read_file(a.brp2025).to_crs(RD)
    points = grass.copy()
    points["geometry"] = grass.representative_point()
    joined = gpd.sjoin(points, later[["gewas", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated()].sort_index()
    grass["next"] = joined["gewas"].values
    grass = grass[grass["next"].notna()].reset_index(drop=True)
    grass["converted"] = (~grass["next"].str.lower().str.contains("grasland")).astype(int)
    grass.to_crs(4326).to_file("grass2024.geojson", driver="GeoJSON")
    print(f"{len(grass)} parcels, {int(grass.converted.sum())} no longer grassland in 2025")


if __name__ == "__main__":
    main()
