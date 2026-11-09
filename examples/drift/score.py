# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Score the drift flags of labels.h5 against the changes recorded in grass2024.geojson."""
import geopandas as gpd
import h5py
import numpy as np

from s2cube.labels import flag_drift


def report(flag, truth):
    tp, fp = int((flag & truth).sum()), int((flag & ~truth).sum())
    fn, tn = int((~flag & truth).sum()), int((~flag & ~truth).sum())
    return f"tp {tp} fp {fp} fn {fn} tn {tn} precision {tp / max(tp + fp, 1):.3f} recall {tp / max(tp + fn, 1):.3f}"


def main():
    truth = gpd.read_file("grass2024.geojson").converted.values.astype(bool)
    with h5py.File("labels.h5", "r") as f:
        years = [str(p) for p in f.attrs["periods"]]
        values, flags = f["drift_index"][:], f["drift_flag"][:]
    first, last = years.index("2024"), years.index("2025")
    ok = ~np.isnan(values).any(axis=0)
    print(f"{int(ok.sum())} parcels with data, {int(truth[ok].sum())} changed")
    print("default, 2025:", report(flags[last][ok], truth[ok]))
    print(f"default, flagged in 2024 although still grassland: {int(flags[first][ok].sum())}")
    for drop in (0.1, 0.15, 0.2, 0.3):
        print(f"drift_drop {drop}:", report(flag_drift(values, drop, 0.4)[last][ok], truth[ok]))


if __name__ == "__main__":
    main()
