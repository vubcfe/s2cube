# Installation

## Requirements

* Python 3.9 - 3.12 on Linux, macOS or Windows.
* Dependencies (installed automatically): `numpy`, `scipy`, `h5py`, `rasterio >= 1.3`
  (ships GDAL and PROJ in its wheels), `shapely >= 2`, `requests`, `pyyaml`, `pillow`.
* Optional: `pyogrio` or `fiona` to read vector formats other than GeoJSON (Shapefile,
  GeoPackage, ...); `torch` to feed the datasets to a `DataLoader`.
* Internet access to `planetarycomputer.microsoft.com` or `earth-search.aws.element84.com`.
  No account or key is required.

## pip

```bash
pip install s2cube
```

## From source (development)

```bash
git clone https://github.com/vubcfe/s2cube
cd s2cube
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest                                                  # offline test suite, ~5 s
```

## conda

```bash
conda create -n s2cube -c conda-forge python=3.11 rasterio h5py shapely scipy pyyaml pillow requests
conda activate s2cube
pip install s2cube --no-deps
```

## Check the installation

```bash
s2cube --version
pytest -m network          # optional: queries the real catalogues
```
