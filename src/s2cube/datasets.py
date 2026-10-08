# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cao Vu Bui and the s2cube contributors
"""Framework-agnostic map-style datasets (``__len__`` / ``__getitem__`` returning NumPy arrays).

They work directly with ``torch.utils.data.DataLoader`` without s2cube importing PyTorch.
Each worker should open its own file handle; the datasets therefore open the HDF5 files
lazily on first access.
"""
from __future__ import annotations

import json

import h5py
import numpy as np


class _Lazy:
    def __init__(self, path):
        self.path, self._f = str(path), None

    @property
    def f(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_f"] = None
        return d


class UnlabeledWindows(_Lazy):
    """Unlabelled windows for self-supervised pretraining.

    Item = ``(X, valid)`` with X (T, B, w, w) float32 reflectance (NaN-free; no data = 0) and
    valid (T, w, w) bool from the clear-sky mask.
    """

    def __init__(self, cube_path, clear_scl=(4, 5, 6)):
        super().__init__(cube_path)
        self.clear_scl = clear_scl
        with h5py.File(self.path, "r") as f:
            self.n = f["unlabeled/X"].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        X = self.f["unlabeled/X"][k].astype("f4") / 10000.0
        valid = np.isin(self.f["unlabeled/SCL"][k], self.clear_scl)
        return X, valid


class LabeledBlocks(_Lazy):
    """Labelled blocks of one fold role ("train" / "val" / "test"), optionally restricted to a
    label budget (``ratio``, ``seed``) from ``splits.json``.

    Item = ``(X, valid, y)``: X (T, B, p, p) float32, valid (T, p, p) bool, y (p, p) uint8 with
    255 = ignore, where p = block size. With ``period_labels=True`` y is (P, p, p).
    """

    def __init__(self, cube_path, labels_path, splits_path, fold: int, role: str = "train",
                 ratio: float | None = None, seed: int = 0, clear_scl=(4, 5, 6), period_labels=False):
        super().__init__(cube_path)
        self.labels_path, self.clear_scl, self.period_labels = str(labels_path), clear_scl, period_labels
        self._lf = None
        with h5py.File(self.path, "r") as cf, h5py.File(self.labels_path, "r") as lf:
            if lf.attrs.get("crs") != cf.attrs["crs"]:
                raise ValueError("labels.h5 and cube.h5 use different CRSs; were they built from the same project?")
            for sid in (k for k in lf if isinstance(lf[k], h5py.Group)):
                if sid in cf["sites"] and lf[f"{sid}/label"].shape != cf[f"sites/{sid}/SCL"].shape[1:]:
                    raise ValueError(f"site {sid}: label grid {lf[f'{sid}/label'].shape} does not match the cube "
                                     f"{cf[f'sites/{sid}/SCL'].shape[1:]}; rebuild labels.h5 for this cube")
        sp = json.loads(open(splits_path).read())
        fd = sp["folds"][fold]
        with h5py.File(self.labels_path, "r") as lf:
            self.p = int(lf.attrs["block_px"])
            if role == "train" and ratio is not None:
                keys = sp["subsets"][str(fold)][str(seed)][str(ratio)]
            else:
                keys = sorted(json.loads(lf.attrs["block_positive_fraction"]))
                keys = [b for b in keys if b.split(":")[0] in fd[role]]
            self.items = []
            for key in keys:
                sid, b = key.split(":")
                blk = lf[f"{sid}/block"][:]
                rr, cc = np.nonzero(blk == int(b))
                self.items.append((sid, int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1))

    @property
    def lf(self):
        if self._lf is None:
            self._lf = h5py.File(self.labels_path, "r")
        return self._lf

    def __getstate__(self):
        d = super().__getstate__()
        d["_lf"] = None
        return d

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, r0, c0, r1, c1 = self.items[i]
        X = self.f[f"sites/{sid}/X"][:, :, r0:r1, c0:c1].astype("f4") / 10000.0
        valid = np.isin(self.f[f"sites/{sid}/SCL"][:, r0:r1, c0:c1], self.clear_scl)
        key = "label_period" if self.period_labels else "label"
        y = self.lf[f"{sid}/{key}"][..., r0:r1, c0:c1]
        return X, valid, y
