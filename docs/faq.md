# Limitations and FAQ

**Which product and collections are supported?** Sentinel-2 Level-2A surface reflectance
from Microsoft Planetary Computer (`planetary-computer`, default) and Element 84 Earth Search
(`earth-search`). Both serve the ESA L2A product as cloud-optimised GeoTIFFs. Other STAC
APIs can be added as a `Preset` (search URL, collection id, asset names).

**Can a project span two UTM zones?** Yes. The project uses one CRS (by default the UTM zone
of the sites' centroid); scenes stored in another zone are reprojected on the fly
(nearest neighbour). For very large areas, split the project per zone.

**How large can a project be?** Sites and windows closer than `read_link_m` are grouped and
each group is read as one block per date, so the empty land between distant sites is never
read. Memory per worker is about `bands x pixels x 2` bytes for the largest group (e.g.
10 x 1400 x 2700 x 2 = 76 MB). Lower `read_link_m` if a group becomes too large.

**Are clouds removed?** No. All dates are kept; the SCL layer and per-date clear fractions
(`dates.csv`, `/qa`) let the model or the user decide. This avoids baking a cloud-mask policy
into the dataset.

**Why are 20 m bands not interpolated?** Interpolation creates values that were never
observed and mixes neighbouring fields. Replication keeps the original measurements; models
can learn their own up-sampling.

**When does co-registration fail?** On dates that are cloudy at almost all locations and in
uniform landscapes (dense forest, water) with no gradients. Such dates are reported as
unreliable and left unchanged. At least `min_sites` clear, textured locations are needed.

**Is the `self` reference an absolute reference?** No. It aligns all dates to the median
geometry of the clearest dates, which is what matters for time-series learning. To align to
an external map (e.g. the imagery on which labels were drawn) use `raster` or `xyz`.

**Why can two cubes of the same place differ slightly?** Where MGRS tiles overlap, the two
granules of the same date can differ by up to ~100 DN. s2cube takes tiles in a fixed order
(tile name) and stores the provenance of every date, so the choice is deterministic and
traceable; tools that use the catalogue's response order may pick the other granule.

**Is the label-drift check generic?** It assumes the labelled class has a stable spectral
index in the chosen months (e.g. evergreen perennial crops in the dry season). For seasonal
crops, choose months accordingly or disable it (`labels.drift: false`).
