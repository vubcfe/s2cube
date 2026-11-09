# Testing the label drift check

Grassland parcels of the 2024 Dutch register (BRP) are treated as polygons drawn once and reused in
2025. The 2025 register says which of them really changed.

```bash
python prepare.py ../netherlands/brp.geojson       # needs geopandas and pyarrow
s2cube build project.yaml && s2cube labels project.yaml
python score.py
```

`prepare.py` downloads the 2024 register (893 MB, CC0) from the Fields of the World collection,
https://source.coop/ftw/harmonized-field-data/nl.
