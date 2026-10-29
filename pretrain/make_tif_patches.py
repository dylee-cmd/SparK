# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Crop large scenes (GeoTIFF etc., anything GDAL can open) into patches for SparK pretraining.

For each scene:
    1. compute the 99th percentile value (`--percentile`) of the valid pixels (per band by default)
    2. contrast stretching: clip to [0, p99] and rescale to [0, 1] (float, NOT converted to uint8)
    3. slide a `--patch_size` window with a stride of `--stride` (100 by default) pixels
    4. drop the patch if no-data/shadow pixels (NaN, or all bands == nodata value) are >= `--max_invalid_ratio` (20%)
    5. save the patch as a float GeoTIFF (with the georeference of the patch); 16-bit float by default (`--dtype`), half the size of float32
The saved patches can be used directly by `main.py --data_type=tif --data_path=<dst_dir>` (no normalization).

Usage (run under `pretrain/`):
    python3 make_tif_patches.py --src_dir=/path/to/scenes --dst_dir=/path/to/patches
    python3 make_tif_patches.py --src_dir=/path/to/scenes --dst_dir=/path/to/patches --patch_size=224 --stride=100 --bands 3 2 1 --workers=8
    python3 make_tif_patches.py --src_dir=/path/to/scenes --dst_dir=/path/to/patches --dtype=float32   # full precision, 2x larger
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def find_scenes(src_dir, exts, recursive):
    exts = tuple(e.lower() for e in exts)
    if not recursive:
        return sorted(os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(exts))
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(src_dir) for f in fs if f.lower().endswith(exts))


def read_bands(ds, bands, xoff, yoff, xsize, ysize, buf_xsize=None, buf_ysize=None) -> np.ndarray:
    """ read the window of the given (1-based) bands as a float32 array of shape (C, h, w) """
    return np.stack([
        ds.GetRasterBand(b).ReadAsArray(xoff, yoff, xsize, ysize, buf_xsize=buf_xsize, buf_ysize=buf_ysize).astype(np.float32)
        for b in bands
    ], axis=0)


def invalid_mask(arr: np.ndarray, nodata_values) -> np.ndarray:
    """ (C, h, w) => (h, w); a pixel is invalid (no-data/shadow) if any band is NaN/inf, or all bands equal the nodata value """
    nan = ~np.isfinite(arr).all(axis=0)
    nodata = np.ones(arr.shape[1:], dtype=bool)
    for c, v in enumerate(nodata_values):
        nodata &= arr[c] == v
    return nan | nodata


def compute_upper_values(ds, bands, nodata_values, percentile, mode, max_side):
    """ compute the `percentile`-th value of valid pixels on a (nearest-neighbour) downsampled scene, to save time and memory """
    W, H = ds.RasterXSize, ds.RasterYSize
    ratio = max(1., max(W, H) / max_side)
    arr = read_bands(ds, bands, 0, 0, W, H, buf_xsize=max(1, round(W / ratio)), buf_ysize=max(1, round(H / ratio)))
    valid = ~invalid_mask(arr, nodata_values)
    if not valid.any():
        return None
    if mode == 'band':
        return [float(np.percentile(arr[c][valid], percentile)) for c in range(arr.shape[0])]
    return [float(np.percentile(arr[:, valid], percentile))] * arr.shape[0]


def sliding_positions(length, patch_size, stride):
    pos = list(range(0, length - patch_size + 1, stride))
    if pos and pos[-1] != length - patch_size:  # also cover the right/bottom border
        pos.append(length - patch_size)
    return pos


def process_scene(job):
    path, args = job
    stem = os.path.splitext(os.path.basename(path))[0]
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    W, H, ps = ds.RasterXSize, ds.RasterYSize, args.patch_size
    bands = args.bands or list(range(1, ds.RasterCount + 1))
    nodata_values = []
    for b in bands:
        v = ds.GetRasterBand(b).GetNoDataValue()
        nodata_values.append(args.nodata if (v is None or args.ignore_file_nodata) else v)
    info = dict(scene=path, width=W, height=H, bands=bands, nodata=nodata_values, saved=0, dropped=0)

    xs, ys = sliding_positions(W, ps, args.stride), sliding_positions(H, ps, args.stride)
    if not xs or not ys:
        info['skipped'] = f'smaller than patch_size={ps}'
        return info
    upper = compute_upper_values(ds, bands, nodata_values, args.percentile, args.percentile_mode, args.stat_max_side)
    if upper is None:
        info['skipped'] = 'no valid pixel'
        return info
    info['p_upper'] = upper
    upper = np.maximum(np.array(upper, dtype=np.float32), 1e-12).reshape(-1, 1, 1)

    out_dir = os.path.join(args.dst_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    gt, proj = ds.GetGeoTransform(can_return_null=True), ds.GetProjection()
    options = [f'COMPRESS={args.compress}'] + (['PREDICTOR=3'] if args.compress in ('DEFLATE', 'LZW', 'ZSTD') else [])
    if args.dtype == 'float16':
        # a Float32 band stored with NBITS=16 is written as IEEE half-precision floats, and is read back as float32 by GDAL:
        # half the size of float32; the error is <= 2^-11 (~4.9e-4) in [0, 1], and 0 and 1 are exact
        options.append('NBITS=16')

    for y in ys:
        strip = read_bands(ds, bands, 0, y, W, ps)    # (C, ps, W): read one row of patches at once
        strip_invalid = invalid_mask(strip, nodata_values)
        for x in xs:
            inv = strip_invalid[:, x:x + ps]
            if inv.mean() >= args.max_invalid_ratio:
                info['dropped'] += 1
                continue

            # contrast stretching: [0, p_upper] => [0, 1]
            patch = np.nan_to_num(strip[:, :, x:x + ps], nan=0., posinf=0., neginf=0.)
            patch = np.clip(patch, 0., upper) / upper
            patch[:, inv] = 0.

            dst = driver.Create(os.path.join(out_dir, f'{stem}_y{y:06d}_x{x:06d}.tif'), ps, ps, len(bands), gdal.GDT_Float32, options=options)
            if gt is not None:
                dst.SetGeoTransform((gt[0] + x * gt[1] + y * gt[2], gt[1], gt[2], gt[3] + x * gt[4] + y * gt[5], gt[4], gt[5]))
            if proj:
                dst.SetProjection(proj)
            for c in range(len(bands)):
                dst.GetRasterBand(c + 1).WriteArray(patch[c].astype(np.float32, copy=False))
            dst.FlushCache()
            dst = None
            info['saved'] += 1
    ds = None
    return info


def main():
    parser = argparse.ArgumentParser(description='crop large scenes into [0, 1] float tif patches for SparK pretraining')
    parser.add_argument('--src_dir', type=str, required=True, help='folder containing the large scenes')
    parser.add_argument('--dst_dir', type=str, required=True, help='output folder (a sub-folder is created for each scene)')
    parser.add_argument('--patch_size', type=int, default=224, help='should be >= the --input_size of main.py')
    parser.add_argument('--stride', type=int, default=100, help='sliding stride in pixels')
    parser.add_argument('--percentile', type=float, default=99., help='the pixel value at this percentile is mapped to 1')
    parser.add_argument('--percentile_mode', type=str, default='band', choices=['band', 'global'], help="compute the percentile for each band ('band') or for all bands together ('global')")
    parser.add_argument('--max_invalid_ratio', type=float, default=0.2, help='drop the patch if the ratio of no-data/shadow pixels >= this value')
    parser.add_argument('--nodata', type=float, default=0., help='no-data value, used when the file does not define one')
    parser.add_argument('--ignore_file_nodata', action='store_true', help='always use --nodata instead of the nodata value defined in the file')
    parser.add_argument('--bands', type=int, nargs='+', default=None, help='1-based band indices to keep (default: all bands); SparK uses 1 or 3 bands')
    parser.add_argument('--exts', type=str, nargs='+', default=['.tif', '.tiff'], help='extensions of the scene files')
    parser.add_argument('--no_recursive', action='store_true', help='do not search the scenes in sub-folders of --src_dir')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'],
                        help='storage precision of the [0, 1] values: float16 (half the size, error <= 4.9e-4; read as float32 for training) or float32')
    parser.add_argument('--compress', type=str, default='NONE', choices=['NONE', 'DEFLATE', 'LZW', 'ZSTD'], help='GeoTIFF compression of the patches')
    parser.add_argument('--stat_max_side', type=int, default=4096, help='the scene is downsampled to this longest side to compute the percentile')
    parser.add_argument('--workers', type=int, default=1, help='number of scenes processed in parallel')
    args = parser.parse_args()

    scenes = find_scenes(args.src_dir, args.exts, recursive=not args.no_recursive)
    if args.dst_dir and os.path.abspath(args.dst_dir).startswith(os.path.abspath(args.src_dir) + os.sep) and not args.no_recursive:
        scenes = [s for s in scenes if not os.path.abspath(s).startswith(os.path.abspath(args.dst_dir) + os.sep)]
    print(f'[make_tif_patches] found {len(scenes)} scenes in {args.src_dir}')
    os.makedirs(args.dst_dir, exist_ok=True)

    start, infos = time.time(), []
    jobs = [(s, args) for s in scenes]
    with Pool(args.workers) if args.workers > 1 else _Serial() as pool:
        for i, info in enumerate(pool.imap_unordered(process_scene, jobs)):
            infos.append(info)
            msg = f"skipped ({info['skipped']})" if 'skipped' in info else f"saved={info['saved']}, dropped={info['dropped']}, p{args.percentile:g}={['%.4g' % v for v in info['p_upper']]}"
            print(f"[{i + 1}/{len(scenes)}] {info['scene']} ({info['width']}x{info['height']}): {msg}", flush=True)

    with open(os.path.join(args.dst_dir, 'make_tif_patches_info.json'), 'w') as f:
        json.dump(dict(args=vars(args), scenes=sorted(infos, key=lambda d: d['scene'])), f, indent=2)
    print(f"[make_tif_patches] saved {sum(d['saved'] for d in infos)} patches, dropped {sum(d['dropped'] for d in infos)} patches, "
          f'to {args.dst_dir} ({time.time() - start:.1f}s)')


class _Serial:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def imap_unordered(self, fn, jobs): return map(fn, jobs)


if __name__ == '__main__':
    main()
