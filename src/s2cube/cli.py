# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Command-line interface: ``s2cube <command> config.yaml``.

Typical workflow::

    s2cube plan     project.yaml      # grids and available dates, nothing downloaded
    s2cube build    project.yaml      # download (resumable)
    s2cube coreg    project.yaml      # measure + correct + verify geolocation
    s2cube labels   project.yaml      # pixel labels (+ per-year label drift)
    s2cube splits   project.yaml      # spatial folds + nested label budgets
    s2cube info     project.yaml      # summary and per-date QA table
    s2cube manifest project.yaml      # SHA-256 manifest of the outputs
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Config


def _cfg(a) -> Config:
    return Config.load(a.config)


def cmd_plan(a):
    from . import stac
    from .cube import plan

    cfg = _cfg(a)
    p = plan(cfg)
    print(f"CRS {p.crs.to_string()}  region {p.region.width} x {p.region.height} px "
          f"({p.region.width * cfg.res / 1e3:.1f} x {p.region.height * cfg.res / 1e3:.1f} km)")
    for sid, g in p.sites.items():
        print(f"  site {sid}: {g.width} x {g.height} px")
    print(f"  {len(p.windows)} unlabelled window(s) of {cfg.unlabeled.win_px} px")
    if not a.offline:
        items = stac.search(p.bbox_lonlat, cfg.start, cfg.end, cfg.source, cfg.max_cloud)
        by_date = stac.group_by_date(items, p.crs.to_epsg())
        print(f"{len(items)} scene(s) on {len(by_date)} date(s) from {cfg.source}")


def cmd_build(a):
    from .cube import build

    build(_cfg(a), limit=a.limit)


def cmd_coreg(a):
    from . import coreg

    cfg = _cfg(a)
    path = cfg.path(cfg.output)
    out = path.with_name("coreg_measured.csv")
    rows = coreg.measure(path, cfg.coreg, out_csv=out)
    if a.measure_only:
        coreg.record(path, rows, cfg.coreg)
        return
    fixed = coreg.apply(path, cfg.coreg, rows=rows, backup=not a.no_backup)
    if fixed:
        coreg.verify(path, cfg.coreg, out_csv=path.with_name("coreg_verify.csv"))


def cmd_labels(a):
    from .labels import build_labels

    build_labels(_cfg(a))


def cmd_splits(a):
    from .splits import build_splits

    build_splits(_cfg(a))


def cmd_info(a):
    from . import qa

    cfg = _cfg(a)
    path = cfg.path(cfg.output)
    if a.refinement:
        qa.check_refinement(path)
    print(qa.dumps(qa.summary(path)))
    rows = qa.date_table(path)
    out = path.with_name("dates.csv")
    qa.write_table(rows, out)
    print(f"per-date table -> {out}")


def cmd_manifest(a):
    from . import qa

    cfg = _cfg(a)
    out = cfg.path(cfg.output)
    files = [p for p in (out, out.with_name("labels.h5"), out.with_name("splits.json"),
                         out.with_name("dates.csv")) if p.exists()]
    m = out.with_name("MANIFEST.sha256")
    if a.check:
        bad = qa.check_manifest(m)
        print("all files match" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    qa.write_manifest(files, m)
    print(f"{len(files)} file(s) -> {m}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="s2cube", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=f"s2cube {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("config", help="project YAML file")
        p.set_defaults(fn=fn)
        return p

    add("plan", cmd_plan, "show grids and available dates").add_argument(
        "--offline", action="store_true", help="do not query the catalogue")
    add("build", cmd_build, "download the cube (resumable)").add_argument(
        "--limit", type=int, default=0, help="download at most N dates in this run")
    p = add("coreg", cmd_coreg, "measure, correct and verify geolocation offsets")
    p.add_argument("--measure-only", action="store_true")
    p.add_argument("--no-backup", action="store_true", help="do not keep a copy of the cube before correcting")
    add("labels", cmd_labels, "rasterise labels")
    add("splits", cmd_splits, "spatial folds and nested label budgets")
    add("info", cmd_info, "summary and per-date QA table").add_argument(
        "--refinement", action="store_true", help="also record ESA's geometric refinement flag per date")
    add("manifest", cmd_manifest, "write or check the SHA-256 manifest").add_argument(
        "--check", action="store_true")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
