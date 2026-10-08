# s2cube

`s2cube` builds co-registered, leakage-safe Sentinel-2 L2A time-series datasets for machine
learning from a list of field sites and label polygons.

A project is one YAML file. Seven commands take it from nothing to a documented dataset:

| Step | Command | Output |
|---|---|---|
| 1 | `s2cube plan project.yaml` | grids, number of available dates (no download) |
| 2 | `s2cube build project.yaml` | `cube.h5` (resumable) |
| 3 | `s2cube coreg project.yaml` | offsets measured, corrected and verified in `cube.h5`; `coreg_*.csv` |
| 4 | `s2cube labels project.yaml` | `labels.h5` |
| 5 | `s2cube splits project.yaml` | `splits.json` |
| 6 | `s2cube info project.yaml` | summary, `dates.csv` |
| 7 | `s2cube manifest project.yaml` | `MANIFEST.sha256` |

* [Installation](installation.md)
* [Tutorial](tutorial.md)
* [Configuration reference](configuration.md)
* [Data format](data_format.md)
* [Methods](methods.md)
* [API reference](api.md)
* [Limitations and FAQ](faq.md)
