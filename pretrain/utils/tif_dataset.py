# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
import time
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


_CACHE_VERSION = 1
_CACHE_DIR_NAME = '.spark_cache'


def default_cache_path(folder: str) -> str:
    """ `<folder>/.spark_cache/tif_patch_list.pt` if `folder` is writable, else `~/.cache/spark/tif_patch_list_<hash of folder>.pt` """
    folder = os.path.abspath(folder)
    if os.access(folder, os.W_OK):
        return os.path.join(folder, _CACHE_DIR_NAME, 'tif_patch_list.pt')
    return os.path.join(os.path.expanduser('~'), '.cache', 'spark', f'tif_patch_list_{hashlib.md5(folder.encode()).hexdigest()[:12]}.pt')


def folder_signature(folder: str, exts) -> list:
    """
    a cheap fingerprint of the patch folder WITHOUT listing millions of files: the mtime of each sub-folder (one per scene for
    `make_tif_patches.py`; the mtime of a folder changes when a file is added/removed in it) and the tif files directly in `folder`.
    note: changes deeper than the 1st-level sub-folders are not detected; use `--rebuild_data_cache` in that case.
    """
    sig = []
    with os.scandir(folder) as it:
        for e in it:
            if e.name == _CACHE_DIR_NAME:
                continue
            if e.is_dir(follow_symlinks=True):
                sig.append(('d', e.name, e.stat(follow_symlinks=True).st_mtime_ns))
            elif e.name.lower().endswith(exts):
                sig.append(('f', e.name, 0))
    return sorted(sig)


def scan_tif_files(folder: str, exts) -> list:
    """ list all tif files under `folder` recursively (sorted, relative paths) """
    rel_paths = []
    for r, dirs, fs in os.walk(folder, followlinks=True):
        if r == folder:
            dirs[:] = [d for d in dirs if d != _CACHE_DIR_NAME]
            prefix = ''
        else:
            prefix = r[len(folder) + 1:] + os.sep     # cheaper than `os.path.relpath` for millions of files
        rel_paths.extend(prefix + f for f in fs if f.lower().endswith(exts))
    rel_paths.sort()
    return rel_paths


def build_patch_list(folder: str, exts) -> dict:
    """ scan the folder, and pack the sorted relative paths into two tensors (utf-8 bytes + offsets); also read the band count of each scene """
    t0 = time.time()
    sig = folder_signature(folder, exts)
    rel_paths = scan_tif_files(folder, exts)
    if len(rel_paths) == 0:
        raise FileNotFoundError(f'no tif file found in {folder}')
    print(f'[TifPatchDataset] scanned {len(rel_paths)} tif files in {time.time() - t0:.1f}s')
    
    dir_bands = {}  # band count of the first patch of each folder (i.e., each scene)
    for p in rel_paths:
        d = os.path.dirname(p)
        if d not in dir_bands:
            ds = gdal.Open(os.path.join(folder, p), gdal.GA_ReadOnly)
            dir_bands[d] = (p, ds.RasterCount)
            ds = None
    
    encoded = [p.encode('utf-8') for p in rel_paths]
    del rel_paths
    offsets = torch.zeros(len(encoded) + 1, dtype=torch.int64)
    torch.cumsum(torch.tensor([len(b) for b in encoded], dtype=torch.int64), dim=0, out=offsets[1:])
    path_bytes = torch.frombuffer(bytearray(b''.join(encoded)), dtype=torch.uint8).clone()
    return dict(version=_CACHE_VERSION, root=folder, exts=list(exts), signature=sig, created=time.time(),
                offsets=offsets, path_bytes=path_bytes, dir_bands=dir_bands)


def _load_cache(cache_path: str, folder: str, exts):
    """ return the cached patch list if it exists and is still valid for `folder`, else None """
    if not os.path.isfile(cache_path):
        return None
    try:
        try:
            cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        except TypeError:   # torch < 1.13 has no `weights_only`
            cache = torch.load(cache_path, map_location='cpu')
    except Exception as e:
        print(f'[TifPatchDataset] failed to read the cache {cache_path}: {e}')
        return None
    if cache.get('version') != _CACHE_VERSION or cache.get('root') != folder or list(cache.get('exts', ())) != list(exts):
        return None
    if [tuple(x) for x in cache['signature']] != folder_signature(folder, exts):
        return None
    return cache


def load_or_build_patch_list(folder: str, exts, cache_path: str, rebuild: bool, is_builder: bool, barrier: Callable = lambda: None) -> dict:
    """
    :param is_builder: True on the process that scans the folder and writes the cache (local rank 0 of each node);
                       the other processes wait for the cache file instead of scanning the same folder concurrently
    :param barrier: a (fast) synchronization, called once BEFORE the (slow) scan; the waiting is done by polling the file, so it never hits the DDP timeout
    """
    if rebuild and is_builder and os.path.isfile(cache_path):
        os.remove(cache_path)   # so that the other processes do not load the old one
    barrier()
    
    if is_builder:
        cache = None if rebuild else _load_cache(cache_path, folder, exts)
        if cache is not None:
            print(f'[TifPatchDataset] loaded the patch list from the cache {cache_path} (use --rebuild_data_cache to rescan)')
            return cache
        cache = build_patch_list(folder, exts)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = f'{cache_path}.tmp{os.getpid()}'
            torch.save(cache, tmp)
            os.replace(tmp, cache_path)     # atomic: the other processes never read a partially written file
            print(f'[TifPatchDataset] saved the patch list cache to {cache_path}')
        except OSError as e:
            raise OSError(f'[TifPatchDataset] cannot write the cache {cache_path} ({e}); set a writable path by --data_cache') from e
        return cache
    
    t0, last_msg = time.time(), time.time()
    while True:
        cache = _load_cache(cache_path, folder, exts)
        if cache is not None:
            return cache
        if time.time() - last_msg > 60:
            msg = f'[TifPatchDataset] waiting for the local rank 0 to build the cache {cache_path} ({time.time() - t0:.0f}s)'
            try:
                print(msg, force=True)  # `force`: SparK's print (utils/misc.py) prints only on the master process by default
            except TypeError:
                print(msg)
            last_msg = time.time()
        time.sleep(3)


class TifPatchDataset(Dataset):
    def __init__(self, folder: str, transform: Callable, bands=None, exts=('.tif', '.tiff'),
                 cache_path='', rebuild_cache=False, is_builder=True, barrier: Callable = lambda: None):
        """
        :param folder: the `--dst_dir` of `make_tif_patches.py` (patches are searched recursively, e.g., `folder/<scene>/*.tif`)
        :param bands: 1-based band indices to use (e.g., (1, 2, 3)); None: use all bands (each patch must have 1 or 3 bands)
        :param cache_path: the patch list cache ('': see `default_cache_path`); scanning millions of files is slow, so the list is cached
        :param rebuild_cache: rescan the folder even if the cache is valid
        :param is_builder, barrier: see `load_or_build_patch_list`
        """
        super(TifPatchDataset, self).__init__()
        self.root = os.path.abspath(folder)
        exts = tuple(e.lower() for e in exts)
        self.cache_path = cache_path or default_cache_path(self.root)
        cache = load_or_build_patch_list(self.root, exts, self.cache_path, rebuild_cache, is_builder, barrier)
        
        # the paths are sorted, so every DDP rank gets the same sample order and `DistInfiniteBatchSampler` splits the indices consistently.
        # they are stored compactly as two tensors (utf-8 bytes + offsets), instead of a tuple of python strings:
        # under DDP (`spawn`), every dataloader worker receives the dataset by pickling; tensors are passed by shared memory,
        # while a tuple of strings is copied to each worker (slow startup, and #GPUs x #workers copies in RAM)
        self._offsets, self._path_bytes = cache['offsets'], cache['path_bytes']
        self.bands = tuple(bands) if bands else None
        self.transform = transform
        self.targets = None  # this is self-supervised learning so we don't need labels
        self.check_bands(cache['dir_bands'])
    
    def get_path(self, index: int) -> str:
        s, e = self._offsets[index].item(), self._offsets[index + 1].item()
        return os.path.join(self.root, self._path_bytes[s:e].numpy().tobytes().decode('utf-8'))
    
    def check_bands(self, dir_bands: dict):
        """ fail fast (before training) if the band count does not fit: check the first patch of each folder (i.e., each scene) """
        if self.bands and (len(self.bands) not in (1, 3) or min(self.bands) < 1):
            raise ValueError(f'--tif_bands={",".join(map(str, self.bands))}: should be 1 or 3 band indices, 1-based')
        errors = []
        for rel, n in dir_bands.values():
            p = os.path.join(self.root, rel)
            if self.bands and max(self.bands) > n:
                errors.append(f'{p}: {n} bands, but --tif_bands={",".join(map(str, self.bands))}')
            elif not self.bands and n not in (1, 3):
                errors.append(f'{p}: {n} bands (the CNN takes 1 or 3 bands)')
        if errors:
            raise ValueError('[TifPatchDataset] band mismatch; select the bands by `main.py --tif_bands=1,2,3` (1-based) '
                             'or `make_tif_patches.py --bands 1 2 3`:\n  ' + '\n  '.join(errors[:20]) + (f'\n  ... ({len(errors)} in total)' if len(errors) > 20 else ''))
    
    def __len__(self):
        return self._offsets.numel() - 1
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(tif_loader(self.get_path(index), self.bands))
