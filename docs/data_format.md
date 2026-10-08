# Data format

All products are plain HDF5 / JSON / CSV, readable without s2cube (`h5py`, MATLAB, R `hdf5r`, ...).

## cube.h5

T dates, B bands, S sites, N unlabelled windows.

| Path | Shape, type | Meaning |
|---|---|---|
| `/dates` | (T,) S10 | acquisition dates `YYYY-MM-DD`, ascending |
| `/done` | (T,) bool | date completely written |
| `/scenes` | (T,) str | JSON list of the scenes of each date: id, MGRS tile, EPSG, platform, cloud cover, processing baseline, DN offset, generation time, asset URLs |
| `/platform`, `/cloud_cover`, `/n_tiles` | (T,) | catalogue metadata (cloud cover is that of the whole tile) |
| `/sites/<id>/X` | (T, B, H, W) uint16 | surface reflectance x 10000; 0 = no data |
| `/sites/<id>/SCL` | (T, H, W) uint8 | ESA scene classification; 0 = no data |
| `/sites/<id>` attrs | `x0`, `y1`, `res_m`, `footprint_wkt` | upper-left corner of the grid (m), pixel size, site polygon |
| `/unlabeled/X` | (N, T, B, w, w) uint16 | unlabelled windows |
| `/unlabeled/SCL` | (N, T, w, w) uint8 | |
| `/unlabeled/origin` | (N, 2) float64 | upper-left corner (x, y) of each window |
| `/qa/site_valid`, `/qa/site_clear` | (T, S) float32 | fraction of pixels with data / clear (SCL 4, 5, 6) |
| `/qa/window_clear` | (T, N) float32 | |
| `/coreg/status` | (T,) int8 | 0 not measured or unreliable, 1 within threshold, 2 corrected |
| `/coreg/measured_m` | (T, 2) float32 | (east, north) offset measured before correction |
| `/coreg/shift_m` | (T, 2) float32 | (east, north) shift applied to the image content |
| `/coreg/residual_m` | (T, 2) float32 | offset measured again after correction |

Root attributes: `s2cube_version`, `bands`, `crs`, `res_m`, `source`, `collection`, `region`,
`config` (the full YAML used), `created`.

Pixel (r, c) of a site covers `x0 + c*res .. x0 + (c+1)*res`, `y1 - (r+1)*res .. y1 - r*res`.
All grids share the same origin modulo the coarsest band resolution, so sites and windows can
be compared pixel by pixel.

## labels.h5

| Path | Shape, type | Meaning |
|---|---|---|
| `/<id>/frac` | (C, H, W) float32 | cover fraction of each class |
| `/<id>/label` | (H, W) uint8 | 0 background, class value, 255 ignore |
| `/<id>/label_period` | (P, H, W) uint8 | per-period labels after the drift check (if enabled) |
| `/<id>/drift_index`, `/<id>/drift_flag` | (P, Q) | per period and polygon: median index, flag |
| `/<id>/block` | (H, W) int32 | block id used for label budgets |

Attributes: `classes`, `purity`, `periods`, `block_positive_fraction` (JSON), `pixel_counts` (JSON).

## splits.json

```json
{"k": 5, "folds": [{"fold": 0, "test": ["p01"], "val": ["p02", "p10"], "train": [...], "min_gap_m": 3170.0}, ...],
 "subsets": {"0": {"0": {"0.05": ["p03:12", ...], "0.1": [...], ...}}},
 "checks": {"min_test_train_gap_m": 1630.0, "min_window_site_gap_m": 512.0, "nested": true}}
```

Block ids are `"<site>:<block>"`. `subsets[fold][seed][ratio]` lists training blocks; smaller
budgets are subsets of larger ones.
