# Methods

## Reading and harmonisation

For each date, all MGRS tiles returned by the catalogue are kept (one per tile, the most
recently processed version). Each band is read at its native resolution through a GDAL warped
VRT whose output grid equals the project grid; tiles of another UTM zone are reprojected,
tiles of the project zone are copied without resampling. Tiles are mosaicked first-valid-wins
in a fixed order (project zone first, then by tile name). The ESA DN offset introduced with
processing baseline 04.00 is removed per tile, based on the scene's processing baseline, so that
reflectance = DN / 10000 for all years. Earth Search serves COGs from which the offset has already
been removed (`earthsearch:boa_offset_applied: true`) although their `raster:bands` metadata still
lists an offset of -0.1; s2cube relies on the flag, not on that metadata.
20 m and 60 m bands are replicated onto the 10 m grid. Grid corners are snapped to multiples of
the coarsest band resolution so replication is exact.

## Co-registration

Sentinel-2 dates of the same place can be offset by more than a pixel, which corrupts
time-series features of small fields and edges. For every date and every location (site or
window) at least `min_clear` clear:

1. the mean of B02-B04 is resampled with cubic splines to a `fine_res` grid (2 m), smoothed
   (sigma = 1 fine pixel) and turned into a gradient-magnitude image;
2. the same is done for the reference;
3. the normalised cross-correlation is computed for every integer shift within
   +-`max_shift_m`; the peak is refined to sub-pixel precision with a parabola per axis;
4. locations with a peak below `min_ncc` are discarded; the date's offset is the median over
   locations and its *spread* the median distance of locations to it. The date is reliable if
   at least `min_sites` locations remain and the spread is below `max_spread_m`.

The `self` reference is the per-pixel temporal median of the `self_dates` clearest dates.
Because those dates may themselves be misaligned, the measurement is repeated
(`self_iterations`): reference dates are shifted by their current offset estimate before the
median is taken, and offsets are anchored at the median geometry of the reference dates.

Reliable offsets above `threshold_m` are corrected by reading the original scenes again on a
grid shifted by the offset (bilinear for reflectance, nearest for SCL); the stored data are
never resampled twice; where two tiles meet, each tile's valid area is eroded by one native pixel
so interpolation never uses no-data, and the neighbouring tile fills the gap. The corrected dates are then measured again (`coreg_verify.csv`, `/coreg/residual_m`).
On synthetic scenes the measured offset is within 0.6 m (0.06 pixel) of the true one.

## ESA geometric refinement flag

Since August 2021 ESA refines the geometry of each Sentinel-2 product against the Global
Reference Image (GRI); when the refinement fails, the product is flagged `NOT_REFINED` in its
datastrip metadata and can be displaced by about a pixel or more. `s2cube info --refinement`
reads this flag for every scene (only the first few MB of the metadata file) and stores it in
`/qa/refined` and `dates.csv`. It is an independent check of the co-registration: in the
Flevoland example, the largest measured offsets all fall on `NOT_REFINED` dates. With
`coreg.self_refined_only: true` the self reference is built from refined dates only, which ties
the cube to the GRI frame.

## Labels

Polygons are rasterised on a `supersample`-times finer grid to obtain the cover fraction of
each class in every pixel. A pixel takes a class (or background) if that class covers at
least `purity`; otherwise, where classes overlap, and outside the annotated area, it is
ignored (255). With `annotated: sites` (wall-to-wall labels such as a parcel register) land not
covered by any polygon is background; `s2cube labels` warns when a site is less than half
covered. With `annotated: labels` (field surveys) only pixels inside polygons are labelled.
Sites are split into square blocks of `block_px` pixels. Validation and test use every block
with a labelled pixel; label budgets draw from blocks at least `block_min_labelled` labelled.

**Label drift.** With `drift: true`, for every year and polygon the median of the per-pixel
temporal median of NDVI over clear pixels (SCL 4-5) in `drift_months` is computed. All polygons
of all sites form one population: each year is centred on the median over all polygons
(removing weather effects shared by the region); a
polygon-year is flagged if it lies more than `drift_drop` below the same polygon's best year,
or if its NDVI is below `drift_low`. Pixels covered for at least `1 - purity` by a flagged
polygon are ignored in that year only (`label_period`).

## Spatial splits

Sites whose footprints are closer than `link_m` are merged into clusters (single linkage).
Clusters are assigned to `k` folds greedily, largest first, balancing the number of labelled
pixels. Fold *i* tests on group *i*, validates on group *i+1* and trains on the rest; the
minimum footprint distance between test/validation and training sites is computed and must
be at least `min_gap_m`. For each seed, training blocks are shuffled within quantile strata of
their positive fraction and interleaved; every label budget is a prefix of this order, so
budgets are nested.
