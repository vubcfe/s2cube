# Configuration reference

A project is described by one YAML file. Relative paths are resolved against the folder of
that file. Unknown keys are rejected, so typos fail early. Defaults are shown below.

```yaml
name: s2cube                  # free text, stored in the cube
start: 2023-01-01             # first acquisition date (inclusive)
end: 2023-12-31               # last acquisition date (inclusive)
source: planetary-computer    # planetary-computer | earth-search
max_cloud: null               # optional catalogue filter on scene cloud cover (%); null keeps all
bands: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]   # any of B01..B12, B8A
res: 10                       # output pixel size: 10 or 20 (must divide every band's resolution)
crs: auto                     # auto = UTM zone of the sites, or e.g. EPSG:32648
output: cube.h5
workers: 6                    # dates downloaded in parallel
scenes_from: null             # path of an existing cube: re-use its frozen scene list (exact re-creation)
read_link_m: 2000             # sites/windows closer than this are read as one block per date

sites:                        # labelled sites; their bounding boxes become the site grids
  path: null                  # GeoJSON / GeoJSONL (others with pyogrio or fiona)
  id_field: id                # unique site id attribute
  crs: EPSG:4326              # CRS of the file
  point_size_m: 1000          # point features become squares of this side
  from_labels: false          # instead of a file: tile the label polygons into square sites
  tile_m: 1000                # tile side (multiple of 20 m)
  min_cover: 0.2              # keep tiles at least this fraction covered by label polygons
  max_sites: null             # random (seeded) subset of the tiles
  seed: 0

unlabeled:                    # windows for self-supervised pretraining
  n: 0
  win_px: 64
  buffer_m: 500               # min distance to any labelled site
  aoi: null                   # area to draw from (default: bounding box of the sites or label polygons)
  aoi_crs: EPSG:4326
  seed: 0

labels:
  path: null                  # label polygons
  crs: EPSG:4326
  class_field: null           # class attribute; null = one positive class (1)
  class_map: null             # {attribute value: class 1..254}; without it the attribute must be an integer
  unmapped: ignore            # values not in class_map: ignore (255) | background (0)
  purity: 0.7                 # min cover of one class (or background) to label a pixel
  supersample: 10             # sub-pixels per side for cover fractions
  annotated: sites            # sites = wall-to-wall labels (unlabelled land is background);
                              # labels = sparse labels (only inside polygons); or a file of annotated areas
  block_px: 9                 # sampling unit for label budgets
  block_min_labelled: 0.5     # min labelled fraction of a block to enter label budgets
  drift: false                # per-year check of label validity
  drift_months: null          # e.g. [1, 2, 3, 4]; null = all months
  drift_drop: 0.15            # flag a polygon-year this far below its best year (after centring)
  drift_low: 0.4              # ... or with an index below this; null disables

splits:
  k: 5                        # folds (>= 3)
  link_m: 1000                # sites closer than this are kept together
  min_gap_m: 1000             # required test/val to train gap (checked)
  ratios: [0.05, 0.1, 0.5, 1.0]
  seeds: [0, 1, 2]
  strata: 4                   # quantile strata of positive fraction for budget sampling

coreg:
  reference: self             # self | raster | xyz
  reference_path: null        # raster reference (any GDAL format)
  xyz_url: null               # template with {z}, {x}, {y}, e.g. https://tiles.example.org/{z}/{x}/{y}.png
  xyz_zoom: 17
  fine_res: 2.0               # matching grid (m)
  max_shift_m: 30
  min_clear: 0.9              # use a location on a date only if this fraction is clear
  min_ncc: 0.3                # discard weak correlation peaks
  min_sites: 3                # a date is reliable with at least this many agreeing locations
  max_spread_m: 3.0           # ... whose median deviation from the date's offset is below this
  threshold_m: 3.0            # correct reliable offsets larger than this
  self_dates: 15              # clearest dates forming the self reference
  self_iterations: 4          # alignment passes of the self reference
  self_refined_only: false    # self reference from ESA GRI-refined dates only (run `s2cube info --refinement` first)
```
