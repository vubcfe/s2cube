# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Project configuration: one YAML file describes a whole dataset build.

Relative paths inside the YAML file are resolved against the file's own directory, so a
project folder can be moved or shared as a unit. See ``docs/configuration.md`` for every key.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

DEFAULT_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


@dataclass
class SitesConfig:
    path: str | None = None          # vector file of labelled sites (polygons or points)
    id_field: str = "id"
    crs: str = "EPSG:4326"           # CRS of the file (GeoJSON is lon/lat by specification)
    point_size_m: float = 1000.0     # side of the square site drawn around point features
    from_labels: bool = False        # instead of a file: tile the label polygons into square sites
    tile_m: float = 1000.0           # side of those tiles (multiple of 60 m recommended)
    min_cover: float = 0.2           # keep tiles whose area is at least this fraction labelled
    max_sites: int | None = None     # keep a random (seeded) subset of the tiles
    seed: int = 0


@dataclass
class UnlabeledConfig:
    n: int = 0                       # number of random windows for self-supervised pretraining
    win_px: int = 64
    buffer_m: float = 500.0          # minimum distance between windows and labelled sites
    aoi: str | None = None           # region to draw windows from (default: sites' bounding box)
    aoi_crs: str = "EPSG:4326"
    seed: int = 0


@dataclass
class LabelsConfig:
    path: str | None = None          # polygons of the labelled class(es)
    crs: str = "EPSG:4326"
    class_field: str | None = None   # class attribute; None = single (positive) class
    class_map: dict | None = None    # {attribute value: class 1..254}, e.g. {Maize: 1, Cassava: 2}
    unmapped: str = "ignore"         # polygons whose value is not in class_map: "ignore" | "background"
    purity: float = 0.7              # min fraction of one class for a pixel to be labelled
    supersample: int = 10            # sub-pixels per side when computing cover fractions
    annotated: str = "sites"         # "sites" = whole site annotated (wall-to-wall labels);
                                     # "labels" = only inside label polygons (sparse labels); or a file
    block_px: int = 9                # block size for label-ratio sampling
    block_min_labelled: float = 0.5  # blocks entering label budgets must be this fraction labelled
    drift: bool = False              # flag polygons whose index departs from the others per year
    drift_months: list | None = None # months used for the drift check (e.g. dry season)
    drift_drop: float = 0.15
    drift_low: float | None = 0.4


@dataclass
class SplitsConfig:
    k: int = 5
    link_m: float = 1000.0           # sites closer than this end up in the same cluster/fold
    min_gap_m: float = 1000.0        # required gap between test/val sites and train sites
    ratios: list = field(default_factory=lambda: [0.05, 0.1, 0.5, 1.0])
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    strata: int = 4


@dataclass
class CoregConfig:
    reference: str = "self"          # "self" | "raster" | "xyz"
    reference_path: str | None = None
    xyz_url: str | None = None       # template with {z}, {x}, {y}; check the provider's terms
    xyz_zoom: int = 17
    fine_res: float = 2.0            # matching grid (m)
    max_shift_m: float = 30.0
    min_clear: float = 0.9           # a site is used on a date only if this fraction is clear
    min_ncc: float = 0.3
    min_sites: int = 3
    max_spread_m: float = 3.0
    threshold_m: float = 3.0         # correct dates whose reliable shift exceeds this
    self_dates: int = 15             # clearest dates averaged into the "self" reference
    self_iterations: int = 4         # re-align reference dates and re-measure (self reference)
    self_refined_only: bool = False  # build the self reference only from dates ESA refined against
                                     # the Global Reference Image (needs `s2cube info --refinement`)


@dataclass
class Config:
    name: str = "s2cube"
    start: str = "2023-01-01"
    end: str = "2023-12-31"
    source: str = "planetary-computer"
    max_cloud: float | None = None
    bands: list = field(default_factory=lambda: list(DEFAULT_BANDS))
    res: float = 10.0
    crs: str = "auto"                # "auto" = UTM zone of the sites' centroid
    output: str = "cube.h5"
    workers: int = 6
    scenes_from: str | None = None   # re-use the scene list frozen in another cube (exact re-creation)
    read_link_m: float = 2000.0      # sites/windows closer than this are read as one block per date
    sites: SitesConfig = field(default_factory=SitesConfig)
    unlabeled: UnlabeledConfig = field(default_factory=UnlabeledConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    coreg: CoregConfig = field(default_factory=CoregConfig)
    base_dir: str = "."

    _SECTIONS = {"sites": SitesConfig, "unlabeled": UnlabeledConfig, "labels": LabelsConfig,
                 "splits": SplitsConfig, "coreg": CoregConfig}

    @classmethod
    def from_dict(cls, d: dict, base_dir=".") -> Config:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        kw = {}
        for k, v in d.items():
            if k in cls._SECTIONS:
                sec = cls._SECTIONS[k]
                bad = set(v or {}) - {f.name for f in fields(sec)}
                if bad:
                    raise ValueError(f"unknown keys in '{k}': {sorted(bad)}")
                kw[k] = sec(**(v or {}))
            else:
                kw[k] = v
        for k in ("start", "end"):
            if k in kw:
                kw[k] = str(kw[k])
        cfg = cls(**kw)
        cfg.base_dir = str(Path(base_dir).resolve())
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path) -> Config:
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()), base_dir=path.parent)

    def path(self, p: str | None) -> Path | None:
        """Resolve a path from the configuration relative to the configuration file."""
        if p is None:
            return None
        q = Path(p).expanduser()
        return q if q.is_absolute() else Path(self.base_dir) / q

    def validate(self) -> None:
        from .stac import NATIVE_RES, PRESETS

        if self.sites.from_labels and not self.labels.path:
            raise ValueError("sites.from_labels needs labels.path")
        if self.sites.from_labels and self.sites.tile_m % 20:
            raise ValueError("sites.tile_m must be a multiple of 20 m")
        if self.labels.unmapped not in ("ignore", "background"):
            raise ValueError("labels.unmapped must be 'ignore' or 'background'")
        if self.labels.class_map and not self.labels.class_field:
            raise ValueError("labels.class_map needs labels.class_field")
        if self.labels.class_map and any(not 1 <= int(v) <= 254 for v in self.labels.class_map.values()):
            raise ValueError("labels.class_map values must be integers in 1..254")

        bad = [b for b in self.bands if b not in NATIVE_RES or b == "SCL"]
        if bad:
            raise ValueError(f"unknown bands {bad}")
        if self.source not in PRESETS:
            raise ValueError(f"unknown source {self.source!r}; choose from {sorted(PRESETS)}")
        if self.res not in (10, 20) or any(NATIVE_RES[b] % self.res for b in self.bands):
            raise ValueError("res must be 10 or 20 m (the 20 m SCL layer is always read) and divide "
                             "every band's native resolution")
        if self.start > self.end:
            raise ValueError("start is after end")
        if not 0.5 <= self.labels.purity <= 1:
            raise ValueError("labels.purity must be in [0.5, 1]")
        if self.coreg.reference not in ("self", "raster", "xyz"):
            raise ValueError("coreg.reference must be 'self', 'raster' or 'xyz'")
        if self.splits.k < 3:
            raise ValueError("splits.k must be >= 3 (test, validation and training folds)")
        if not all(0 < r <= 1 for r in self.splits.ratios):
            raise ValueError("splits.ratios must lie in (0, 1]")

    def to_yaml(self) -> str:
        d = asdict(self)
        d.pop("base_dir")
        return yaml.safe_dump(d, sort_keys=False)
