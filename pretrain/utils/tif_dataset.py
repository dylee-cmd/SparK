# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset

gdal.UseExceptions()


def tif_loader(path) -> torch.Tensor:
    """ read a [0, 1] float tif patch (made by `make_tif_patches.py`) as a (3, H, W) float32 tensor, WITHOUT any normalization """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    arr = ds.ReadAsArray()
    ds = None
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0., posinf=0., neginf=0.)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[0] == 1:   # single band (e.g., SAR or panchromatic): repeat to 3 channels
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] != 3:
        raise ValueError(f'{path} has {arr.shape[0]} bands, but the CNN takes 1 or 3 bands; use `make_tif_patches.py --bands` to select them')
    return torch.from_numpy(np.ascontiguousarray(arr))


class Clamp(torch.nn.Module):
    """ keep the values in [0, 1] after the bicubic interpolation of RandomResizedCrop (bicubic can overshoot) """
    def __init__(self, lo=0., hi=1.):
        super(Clamp, self).__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x: torch.Tensor):
        return x.clamp_(self.lo, self.hi)

    def __repr__(self):
        return f'{type(self).__name__}(lo={self.lo}, hi={self.hi})'


class TifPatchDataset(Dataset):
    def __init__(self, folder: str, transform: Callable, exts=('.tif', '.tiff')):
        super(TifPatchDataset, self).__init__()
        self.samples = tuple(sorted(
            os.path.join(r, f) for r, _, fs in os.walk(folder, followlinks=True) for f in fs if f.lower().endswith(exts)
        ))
        if len(self.samples) == 0:
            raise FileNotFoundError(f'no tif file found in {folder}')
        self.transform = transform
        self.targets = None  # this is self-supervised learning so we don't need labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(tif_loader(self.samples[index]))
