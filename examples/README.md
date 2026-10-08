# Examples

| Folder | Region, period | Labels | Shows | Size / time |
|---|---|---|---|---|
| `minimal/` | Flevoland (NL), May-June 2025 | BRP parcels (public domain), 4 classes | the whole workflow in minutes | ~30 dates, ~3 min |
| `netherlands/` | Flevoland (NL), 2025 | BRP parcels, 8 crop classes, wall-to-wall | multi-class labels, sites tiled from labels, spatial folds | ~170 dates |
| `kenya/` | Busia/Siaya (Kenya), 2019 | PlantVillage field survey (CC BY-SA 4.0), 5 classes, sparse | persistent cloud, sparse labels, pre-2022 processing baseline | ~73 dates |

Each folder has a `get_labels.py` that downloads the reference polygons from their publisher
(they are not redistributed here), and a `project.yaml`:

```bash
cd examples/netherlands
python get_labels.py
s2cube build project.yaml && s2cube coreg project.yaml && s2cube labels project.yaml \
  && s2cube splits project.yaml && s2cube info project.yaml && s2cube manifest project.yaml
```

Label sources:

* RVO (2025). *Basisregistratie Gewaspercelen (BRP)*. PDOK OGC API Features,
  https://api.pdok.nl/rvo/gewaspercelen/ogc/v1 (public domain).
* PlantVillage (2019). *PlantVillage Kenya Ground Reference Crop Type Dataset*. Radiant MLHub,
  https://doi.org/10.34911/rdnt.u41j87 (CC BY-SA 4.0).
