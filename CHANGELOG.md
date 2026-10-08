# Changelog

All notable changes are documented here. The project follows [Semantic Versioning](https://semver.org).

## [0.1.0] - 2026-10-08

First public release.

- `plan`, `build` (resumable, scene provenance frozen at creation), `coreg` (measure, correct,
  verify; `self`, `raster` and `xyz` references; iterative self reference), `labels` (cover
  fractions, purity, ignore pixels, per-year label drift), `splits` (spatial clusters, gap
  check, nested label budgets), `info`, `manifest`.
- STAC presets for Microsoft Planetary Computer and Element 84 Earth Search.
- NumPy datasets `UnlabeledWindows` and `LabeledBlocks` usable with PyTorch.
- Offline test suite on synthetic GeoTIFF scenes; network tests against the live catalogues.
- `s2cube info --refinement` records ESA's geometric refinement flag per date;
  `coreg.self_refined_only` builds the self reference from GRI-refined dates.
- `scenes_from` re-creates a dataset on exactly the scenes recorded in another cube.
- Sites and windows are read in spatial groups (`read_link_m`), not as one bounding box.
