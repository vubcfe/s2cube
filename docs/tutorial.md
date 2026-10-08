# Tutorial

This tutorial runs `examples/minimal`: four 600 m sites on arable land in Flevoland, the
Netherlands, May-June 2025, labelled with the public Dutch parcel register (BRP). It needs
~150 MB of downloads and about three minutes.

## 1. Describe the project

```bash
cd examples/minimal
python ../netherlands/get_labels.py --bbox 5.60 52.50 5.68 52.55 --out brp.geojson
```

`brp.geojson` holds ~2000 parcels with their crop (`gewas`). With `sites.from_labels: true`
the parcels are tiled into 600 m squares, and four tiles that are at least 60 % agricultural
are drawn at random (seeded). `labels.class_map` maps four crops to classes 1-4; other crops
are ignored (255) and land outside parcels is background (0). See the
[configuration reference](configuration.md) for every option.

## 2. Plan

```text
$ s2cube plan project.yaml
CRS EPSG:32631  region 628 x 598 px (6.3 x 6.0 km)
  site E0677p40_N5821p80: 60 x 60 px
  ...
  8 unlabelled window(s) of 32 px
35 scene(s) on 29 date(s) from planetary-computer
```

Site ids encode the lower-left corner of the tile in km (UTM).

## 3. Build

```text
$ s2cube build project.yaml
29 date(s) to read, region 628 x 598 px
  [   1/29] 2025-05-01  valid 100.0%  clear  99.6%  ~4 min left
  ...
```

The command can be interrupted and run again; it continues with the dates not yet written,
using exactly the scenes recorded at creation.

## 4. Co-register

```text
$ s2cube coreg project.yaml
  pass 1: largest change of a reference offset 12.29 m
  ...
  2025-05-11  n= 12  E  -9.64  N  +7.24 m  spread  0.45
  ...
1 date(s) to read
  2025-05-11  n= 12  E  +0.20  N  -0.50 m  spread  0.30
```

One date is displaced by 12 m (more than one pixel); all twelve locations agree within
0.45 m. It is read again on a shifted grid; the verification pass finds 0.5 m left.
`coreg_measured.csv` and `coreg_verify.csv` keep the numbers, the original cube is kept as
`cube.orig.h5`. Cloudy dates are reported as unreliable and left untouched.

## 5. Labels, splits, report

```bash
s2cube labels project.yaml      # labels.h5
s2cube splits project.yaml      # splits.json
s2cube info project.yaml        # summary + dates.csv
s2cube manifest project.yaml    # MANIFEST.sha256
```

## 6. Use in Python / PyTorch

```python
from torch.utils.data import DataLoader
from s2cube.datasets import LabeledBlocks

ds = LabeledBlocks("cube.h5", "labels.h5", "splits.json", fold=0, role="train", ratio=0.5, seed=0)
for X, valid, y in DataLoader(ds, batch_size=8, num_workers=2):
    ...   # X (8, T, B, 10, 10), valid (8, T, 10, 10), y (8, 10, 10) with 255 = ignore
```
