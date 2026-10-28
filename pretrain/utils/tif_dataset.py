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


def tif_loader(path, bands=None) -> torch.Tensor:
    """
    read a [0, 1] float tif patch (made by `make_tif_patches.py`) as a (3, H, W) float32 tensor, WITHOUT any normalization
    :param bands: 1-based band indices to read (e.g., (1, 2, 3)); None: read all bands
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if bands:
        arr = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in bands], axis=0)
    else:
        arr = ds.ReadAsArray()
    ds = None
    arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0., posinf=0., neginf=0.)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[0] == 1:   # single band (e.g., SAR or panchromatic): repeat to 3 channels
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] != 3:
        raise ValueError(f'{path}: got {arr.shape[0]} bands, but the CNN takes 1 or 3 bands; select them by `main.py --tif_bands` or `make_tif_patches.py --bands`')
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
    def __init__(self, folder: str, transform: Callable, bands=None, exts=('.tif', '.tiff')):
        """
        :param folder: the `--dst_dir` of `make_tif_patches.py` (patches are searched recursively, e.g., `folder/<scene>/*.tif`)
        :param bands: 1-based band indices to use (e.g., (1, 2, 3)); None: use all bands (each patch must have 1 or 3 bands)
        """
        super(TifPatchDataset, self).__init__()
        # sorted: every DDP rank gets the same sample order, so `DistInfiniteBatchSampler` splits the indices consistently
        self.samples = tuple(sorted(
            os.path.join(r, f) for r, _, fs in os.walk(folder, followlinks=True) for f in fs if f.lower().endswith(exts)
        ))
        if len(self.samples) == 0:
            raise FileNotFoundError(f'no tif file found in {folder}')
        self.bands = tuple(bands) if bands else None
        self.transform = transform
        self.targets = None  # this is self-supervised learning so we don't need labels
        self.check_bands()
    
    def check_bands(self):
        """ fail fast (before training) if the band count does not fit: check the first patch of each folder (i.e., each scene) """
        if self.bands and (len(self.bands) not in (1, 3) or min(self.bands) < 1):
            raise ValueError(f'--tif_bands={",".join(map(str, self.bands))}: should be 1 or 3 band indices, 1-based')
        first_of_each_dir = {}
        for p in self.samples:
            first_of_each_dir.setdefault(os.path.dirname(p), p)
        errors = []
        for p in first_of_each_dir.values():
            ds = gdal.Open(p, gdal.GA_ReadOnly)
            n = ds.RasterCount
            ds = None
            if self.bands and max(self.bands) > n:
                errors.append(f'{p}: {n} bands, but --tif_bands={",".join(map(str, self.bands))}')
            elif not self.bands and n not in (1, 3):
                errors.append(f'{p}: {n} bands (the CNN takes 1 or 3 bands)')
        if errors:
            raise ValueError('[TifPatchDataset] band mismatch; select the bands by `main.py --tif_bands=1,2,3` (1-based) '
                             'or `make_tif_patches.py --bands 1 2 3`:\n  ' + '\n  '.join(errors[:20]) + (f'\n  ... ({len(errors)} in total)' if len(errors) > 20 else ''))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(tif_loader(self.samples[index], self.bands))
