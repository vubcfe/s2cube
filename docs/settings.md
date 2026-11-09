# Recommended settings

These are starting points. The case studies measured them in two agricultural regions, so check
them for a new area.

## Small fields and mixed pixels

Bands at 20 m are replicated 2 x 2 onto the 10 m grid. This avoids interpolated values but adds no
detail: one 20 m pixel covers four 10 m pixels. In fields only a few 10 m pixels wide, these bands
mix neighbouring surfaces.

* Raise `labels.purity` (for example to 0.9) so that mixed pixels are ignored. This costs few pixels
  in large parcels (about 3.5 % per step in the Flevoland example) and more in small fields.
* Use `bands: [B02, B03, B04, B08]` if the task does not need the red-edge and shortwave bands.
* Choose `labels.block_px` no smaller than the neighbourhood your model reads.

## Clouds

s2cube stores the Scene Classification Layer and does not mask clouds, so that the policy stays
with the user. `LabeledBlocks` and `UnlabeledWindows` return a clear-sky mask with every item.

* Pixel time series: drop or interpolate observations outside the mask, or use monthly composites
  of clear observations, as in `examples/downstream`.
* Use `max_cloud` to skip very cloudy scenes at build time. This shortens the build, but a scene
  that is cloudy overall can still be clear over your sites.
* Dates without a co-registration measurement are kept and marked in `/coreg/status`. They are
  mostly too cloudy to matter: in the examples they hold 4.6 % and 0.7 % of all clear observations.
  Check `dates.csv` for your project, and exclude these dates if your task is sensitive to geometry.

## Spatial distance thresholds

`splits.link_m` and `splits.min_gap_m` should exceed the range of spatial autocorrelation.

* `examples/downstream/side_experiments.py --only correlogram` measures correlation and label
  agreement against distance. In the examples, band correlation reaches zero within 200-500 m
  (Flevoland) and 500-1000 m (Kenya), which supports the default of 1000 m there.
* Larger fields and larger landscape patterns need larger distances.
* If sites form one contiguous area, use `strategy: patches`. Its guard is only a few blocks wide, so
  it separates the data less than site-level folds do. Prefer site-level folds whenever you can.

## Co-registration

The default threshold of 3 m corrects offsets that clearly exceed the measurement spread. In the
examples, correction improved agreement with parcel boundaries but not classification accuracy.
`cube.orig.h5` is kept, so compare both cubes on your own task. See `examples/downstream`.

## Label drift

The drift check suits classes with a stable spectral index in the chosen months, such as grassland
or perennial crops. It does not suit annual crops in rotation. In the Flevoland test it was
precise (0.98) but incomplete (recall 0.62), so treat it as a conservative filter.
