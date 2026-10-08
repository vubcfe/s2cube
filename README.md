# s2cube

**Build co-registered, leakage-safe Sentinel-2 time-series datasets for machine learning.**

[![tests](https://github.com/vubcfe/s2cube/actions/workflows/tests.yml/badge.svg)](https://github.com/vubcfe/s2cube/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23240761.svg)](https://doi.org/10.5281/zenodo.23240761)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

`s2cube` turns a list of field sites and label polygons into an analysis-ready Sentinel-2 L2A
time series, stored in one HDF5 file, together with pixel labels and spatially separated
cross-validation splits. It is written for researchers who train (self-)supervised models on
satellite image time series and need the dataset to be **correct**, **reproducible** and
**free of spatial leakage**:

| Problem in practice | What s2cube does |
|---|---|
| Dates of the same place are misaligned by 3-15 m (one or more 10 m pixels) | measures the offset of every date by gradient cross-correlation on many locations, corrects reliable offsets by re-reading the original scenes on a shifted grid, and verifies the result |
| 20 m bands are interpolated to 10 m, values change across ESA processing baselines | 20 m bands are replicated 2 x 2 (never interpolated); DN offsets are removed per scene |
| Re-running a download months later silently gives different scenes | dates and exact scene URLs are frozen into the file at creation; downloads resume from that record ; `scenes_from` re-creates a dataset on the same scenes |
| Some products were not geometrically refined by ESA | `s2cube info --refinement` records ESA's GRI refinement flag per date |
| Labels drawn on one year's imagery are reused for other years | per-polygon, per-year index check flags land-use change; those pixels are ignored only in the affected year |
| Random splits put neighbouring pixels in train and test | sites closer than a linkage distance stay in one fold; every fold is checked for a minimum test/train gap; nested label budgets (5 % in 10 % in 50 % ...) |
| Unlabelled data for pretraining overlaps labelled sites | random windows are drawn at a guaranteed distance from every labelled site |

## Installation

```bash
pip install s2cube                       # from PyPI (after release)
# or, from source
git clone https://github.com/vubcfe/s2cube && cd s2cube
pip install -e ".[test]"
```

Requirements: Python >= 3.9, and the wheels of `numpy`, `scipy`, `h5py`, `rasterio` (bundles
GDAL/PROJ), `shapely`, `requests`, `pyyaml`, `pillow`. Linux, macOS and Windows are supported.
No account or API key is needed: data are read from the public STAC catalogues of Microsoft
Planetary Computer (default) or Element 84 Earth Search on AWS.

## Quick start

```bash
cd examples/minimal               # 4 crop sites in Flevoland (NL), May-June 2025
python ../netherlands/get_labels.py --bbox 5.60 52.50 5.68 52.55 --out brp.geojson
s2cube plan   project.yaml        # grids + available dates (nothing downloaded)
s2cube build  project.yaml        # ~3 min, resumable
s2cube coreg  project.yaml        # measure -> correct -> verify geolocation
s2cube labels project.yaml        # labels.h5
s2cube splits project.yaml        # splits.json
s2cube info   project.yaml        # summary + dates.csv
s2cube manifest project.yaml      # MANIFEST.sha256
```

In Python:

```python
from s2cube import Cube, reflectance
from s2cube.datasets import LabeledBlocks, UnlabeledWindows

with Cube("cube.h5") as c:
    print(c.dates[:3], c.bands, c.sites)
    X, scl = c.site(c.sites[0])            # (T, B, H, W) uint16, (T, H, W) uint8
    r = reflectance(X)                     # float32 reflectance, NaN = no data

train = LabeledBlocks("cube.h5", "labels.h5", "splits.json", fold=0, role="train", ratio=0.1, seed=0)
X, valid, y = train[0]                     # works with torch.utils.data.DataLoader
pretrain = UnlabeledWindows("cube.h5")
```

## Outputs

| File | Content |
|---|---|
| `cube.h5` | `/sites/<id>/X (T,B,H,W)`, `/sites/<id>/SCL`, `/unlabeled/X (N,T,B,w,w)`, dates, per-date scene provenance, QA fractions, `/coreg/*` |
| `labels.h5` | per site: class cover fractions, hard labels (255 = ignore), per-year labels and drift flags, block ids |
| `splits.json` | folds (test/val/train sites, measured gaps), nested label budgets per seed, leakage checks |
| `dates.csv` | per date: platform, cloud cover, observed valid/clear fraction, co-registration status and offsets |
| `MANIFEST.sha256` | checksums of all products (`sha256sum -c` compatible) |

The full layout is documented in [docs/data_format.md](docs/data_format.md).

## Documentation

* [Installation](docs/installation.md)
* [Tutorial](docs/tutorial.md)
* [Configuration reference](docs/configuration.md)
* [Data format](docs/data_format.md)
* [Methods](docs/methods.md) (co-registration, label drift, spatial splits)
* [API reference](docs/api.md)
* [Limitations and FAQ](docs/faq.md)

## Tests

```bash
pytest                      # offline: synthetic scenes written as local GeoTIFFs (~5 s)
pytest -m network           # also query the real STAC catalogues (needs internet)
```

The offline suite builds a complete dataset from synthetic scenes with a known displacement
and checks, among others, exact pixel values, the recovery and correction of the displacement,
labels, fold separation and nested subsets.

## Citing

If you use s2cube, please cite the software paper (see [CITATION.cff](CITATION.cff)).

## License

MIT, see [LICENSE.txt](LICENSE.txt). Sentinel-2 data are provided by the European Union's Copernicus
programme under its free and open data policy. When using a web-map tile service as
co-registration reference (`coreg.reference: xyz`), respect that provider's terms of use.
