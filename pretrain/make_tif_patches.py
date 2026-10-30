# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Crop large scenes (SAR L1-D GeoTIFF etc., anything GDAL can open) into patches for SparK pretraining,
with GDAL Info inspection & geometric/radiometric valid-region masking.

For each scene:
    1. find the valid region (bounding box + convex hull of the valid swath) on a downsampled scene via `gdal.Info` inspection;
       invalid (shadow/background) pixels: NaN/inf, all bands == nodata, all bands <= `--min_valid_val`, or masked by the GDAL MaskBand
    2. compute the 99th percentile value (`--percentile`) of the valid pixels (per band by default)
    3. contrast stretching: clip to [0, p99] and rescale to [0, 1] (float, NOT converted to uint8)
    4. slide a `--patch_size` window with a stride of `--stride` (100 by default) pixels, only inside the valid bounding box
    5. drop the patch if it is outside the convex hull by >= `--max_invalid_ratio` (20%), if its invalid pixels are >= `--max_invalid_ratio`,
       or if it is flat (std of the valid pixels < `--min_patch_std`)
    6. save the patch as a float GeoTIFF (with the georeference of the patch); 16-bit float by default (`--dtype`), half the size of float32
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
from osgeo import gdal, ogr

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


def read_gdal_mask(ds, bands, xoff, yoff, xsize, ysize, buf_xsize=None, buf_ysize=None) -> np.ndarray:
    """
    GDAL 내장 MaskBand (PER_DATASET, ALPHA 등)를 읽어 무효(음영) 픽셀 여부를 반환 (True = Invalid).
    """
    b1 = ds.GetRasterBand(bands[0])
    flags = b1.GetMaskFlags()
    # GMF_ALL_VALID(0x01)가 아니면 유효한 마스크 밴드가 존재함
    if not (flags & gdal.GMF_ALL_VALID):
        mask_arr = b1.GetMaskBand().ReadAsArray(
            xoff, yoff, xsize, ysize, buf_xsize=buf_xsize, buf_ysize=buf_ysize
        )
        return mask_arr == 0
    return np.zeros((buf_ysize or ysize, buf_xsize or xsize), dtype=bool)


def inspect_l1d_with_gdalinfo(ds, bands, nodata_values, min_valid_val, max_side=2048):
    """
    gdal.Info(ds, format='json')를 통해 L1-D 영상의 메타데이터, 마스크 플래그, 유효 영역(Footprint/Swath)을 확인하고,
    유효 픽셀이 존재하는 Bounding Box (xmin, ymin, xmax, ymax) 및 저해상도 기하 폴리곤 마스크를 생성합니다.
    """
    ginfo = gdal.Info(ds, format='json', computeMinMax=False, stats=False)
    W, H = ginfo['size'][0], ginfo['size'][1]

    # 1. gdal.Info에서 밴드별 Mask Flags 및 메타데이터 확인
    band_infos = [b for b in ginfo.get('bands', []) if b.get('band') in bands]
    mask_flags = [b.get('maskFlags', []) for b in band_infos]
    metadata = ginfo.get('metadata', {}).get('', {})

    # 2. 메타데이터에 명시적 WKT Footprint가 있는지 탐색 (예: SAR L1-D 표준 메타데이터 키)
    footprint_wkt = None
    for k, v in metadata.items():
        if any(w in k.upper() for w in ('FOOTPRINT', 'POLYGON', 'VALID_REGION', 'BOUNDING_POLYGON')):
            if isinstance(v, str) and ('POLYGON' in v.upper()):
                footprint_wkt = v
                break

    # 3. 다운샘플링 그리드에서 실제 유효 신호 영역(Convex Hull & Bounding Box) 검출
    ratio = max(1.0, max(W, H) / max_side)
    sub_W, sub_H = max(1, round(W / ratio)), max(1, round(H / ratio))
    sub_arr = read_bands(ds, bands, 0, 0, W, H, buf_xsize=sub_W, buf_ysize=sub_H)
    sub_gdal_inv = read_gdal_mask(ds, bands, 0, 0, W, H, buf_xsize=sub_W, buf_ysize=sub_H)

    # 픽셀이 유한하고, nodata가 아니며, min_valid_val(음영/배경 임계값)보다 큰 영역만 유효 신호로 판정
    sub_inv = invalid_mask(sub_arr, nodata_values, min_valid_val=min_valid_val, gdal_inv_mask=sub_gdal_inv)
    sub_valid = ~sub_inv

    if not sub_valid.any():
        return dict(gdal_mask_flags=mask_flags, has_footprint=bool(footprint_wkt), valid_bbox=None, poly_mask_sub=None, ratio=ratio)

    # 유효 픽셀들의 외곽 좌표로부터 OGR Convex Hull 폴리곤을 생성해 L1-D 스트립 내부만 유효 영역으로 확정
    ys_v, xs_v = np.where(sub_valid)
    xmin_s, xmax_s = int(xs_v.min()), int(xs_v.max())
    ymin_s, ymax_s = int(ys_v.min()), int(ys_v.max())

    # 원본 해상도 기준 유효 Bounding Box 산출 (여유 마진 부여)
    xmin = max(0, int(xmin_s * ratio))
    ymin = max(0, int(ymin_s * ratio))
    xmax = min(W, int((xmax_s + 1) * ratio))
    ymax = min(H, int((ymax_s + 1) * ratio))

    # 경계점들을 추출하여 OGR MultiPoint -> ConvexHull 생성 후 저해상도 마스크 래스터화
    # (내부의 작은 호수/음영 홀은 메우면서 L1-D 외곽의 기하학적 삼각형 음영 여백은 깔끔히 제거)
    poly_mask_sub = np.zeros((sub_H, sub_W), dtype=np.uint8)
    pts = ogr.Geometry(ogr.wkbMultiPoint)
    # 각 행(row)별 좌우 끝 유효 픽셀만 샘플링하여 고속으로 Convex Hull 계산
    unique_ys = np.unique(ys_v)
    for y_idx in unique_ys[::max(1, len(unique_ys) // 500)]:
        row_xs = xs_v[ys_v == y_idx]
        for x_idx in (int(row_xs.min()), int(row_xs.max())):
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.AddPoint_2D(float(x_idx) + 0.5, float(y_idx) + 0.5)
            pts.AddGeometry(pt)

    hull = pts.ConvexHull()
    if hull is not None and hull.GetGeometryName() == 'POLYGON':
        mem_drv = gdal.GetDriverByName('MEM')
        mask_ds = mem_drv.Create('', sub_W, sub_H, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform((0, 1, 0, 0, 0, 1))
        ogr_drv = ogr.GetDriverByName('Memory')
        ogr_ds = ogr_drv.CreateDataSource('hull')
        lyr = ogr_ds.CreateLayer('hull', srs=None, geom_type=ogr.wkbPolygon)
        feat = ogr.Feature(lyr.GetLayerDefn())
        feat.SetGeometry(hull)
        lyr.CreateFeature(feat)
        gdal.RasterizeLayer(mask_ds, [1], lyr, burn_values=[1])
        poly_mask_sub = mask_ds.GetRasterBand(1).ReadAsArray()
        feat, lyr, ogr_ds, mask_ds = None, None, None, None
    else:
        poly_mask_sub[ymin_s:ymax_s + 1, xmin_s:xmax_s + 1] = 1

    return dict(
        gdal_mask_flags=mask_flags,
        has_footprint=bool(footprint_wkt),
        valid_bbox=(xmin, ymin, xmax, ymax),
        poly_mask_sub=poly_mask_sub.astype(bool),
        sub_valid=sub_valid,
        sub_arr=sub_arr,
        ratio=ratio
    )


def invalid_mask(arr: np.ndarray, nodata_values, min_valid_val=None, gdal_inv_mask=None) -> np.ndarray:
    """
    (C, h, w) => (h, w)
    픽셀 무효(음영/배경) 판정 조건:
      1) 어느 한 밴드라도 NaN / Inf 인 경우
      2) 모든 밴드가 nodata_values와 일치하는 경우 (부동소수점 오차 감안 np.isclose 적용)
      3) L1-D 음영/배경 여백 제거: 모든 밴드 값이 min_valid_val 이하인 경우 (예: 0 또는 극소 보간 노이즈)
      4) GDAL MaskBand (Alpha / Per-dataset mask)에서 무효(0)로 지정된 경우
    """
    nan = ~np.isfinite(arr).all(axis=0)
    nodata = np.ones(arr.shape[1:], dtype=bool)
    for c, v in enumerate(nodata_values):
        nodata &= np.isclose(arr[c], v, atol=1e-7)

    inv = nan | nodata
    if min_valid_val is not None:
        shadow_or_pad = (arr <= min_valid_val).all(axis=0)
        inv |= shadow_or_pad
    if gdal_inv_mask is not None:
        inv |= gdal_inv_mask
    return inv


def compute_upper_values(sub_arr, sub_valid, percentile, mode):
    """ 이미 inspect 단계에서 읽어들인 유효 픽셀(sub_valid)만을 대상으로 상위 백분위수(p99) 계산 """
    if not sub_valid.any():
        return None
    if mode == 'band':
        return [float(np.percentile(sub_arr[c][sub_valid], percentile)) for c in range(sub_arr.shape[0])]
    return [float(np.percentile(sub_arr[:, sub_valid], percentile))] * sub_arr.shape[0]


def sliding_positions_in_bbox(start, end, patch_size, stride):
    """ 전체 영상이 아닌 gdal.Info 기반 유효 Bounding Box 구간 [start, end) 내에서만 슬라이딩 위치 생성 """
    if end - start < patch_size:
        return []
    pos = list(range(start, end - patch_size + 1, stride))
    if pos and pos[-1] != end - patch_size:  # also cover the right/bottom border
        pos.append(end - patch_size)
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

    # [핵심] gdal.Info 및 다운샘플 스캔을 통해 실제 유효한 L1-D 영역(BBox 및 Convex Hull) 확인
    insp = inspect_l1d_with_gdalinfo(ds, bands, nodata_values, args.min_valid_val, max_side=args.stat_max_side)
    valid_bbox = insp['valid_bbox']

    info = dict(
        scene=path, width=W, height=H, bands=bands, nodata=nodata_values,
        min_valid_val=args.min_valid_val, gdal_mask_flags=insp['gdal_mask_flags'], has_footprint=insp['has_footprint'],
        valid_bbox=valid_bbox, saved=0, dropped=0
    )

    if valid_bbox is None:
        info['skipped'] = 'no valid L1-D pixel found via gdal.Info inspection'
        return info

    xmin, ymin, xmax, ymax = valid_bbox
    xs = sliding_positions_in_bbox(xmin, xmax, ps, args.stride)
    ys = sliding_positions_in_bbox(ymin, ymax, ps, args.stride)
    if not xs or not ys:
        info['skipped'] = f'valid_bbox {valid_bbox} smaller than patch_size={ps}'
        return info

    upper = compute_upper_values(insp['sub_arr'], insp['sub_valid'], args.percentile, args.percentile_mode)
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

    poly_mask_sub = insp['poly_mask_sub']
    sub_H, sub_W = poly_mask_sub.shape
    ratio = insp['ratio']
    bbox_w = xmax - xmin

    for y in ys:
        # 전체 너비 W 대신 유효 BBox 가로 폭(xmin ~ xmax)만 읽어 I/O 속도 향상 (C, ps, bbox_w): read one row of patches at once
        strip = read_bands(ds, bands, xmin, y, bbox_w, ps)
        strip_gdal_inv = read_gdal_mask(ds, bands, xmin, y, bbox_w, ps)
        strip_invalid = invalid_mask(
            strip, nodata_values, min_valid_val=args.min_valid_val, gdal_inv_mask=strip_gdal_inv
        )

        for x in xs:
            rel_x = x - xmin
            # 1) Convex Hull(유효 Swath 폴리곤) 외부 여백인지 고속 체크
            sub_y0 = min(sub_H - 1, int(y / ratio))
            sub_y1 = min(sub_H, max(sub_y0 + 1, int((y + ps) / ratio)))
            sub_x0 = min(sub_W - 1, int(x / ratio))
            sub_x1 = min(sub_W, max(sub_x0 + 1, int((x + ps) / ratio)))
            if (1.0 - poly_mask_sub[sub_y0:sub_y1, sub_x0:sub_x1].mean()) >= args.max_invalid_ratio:
                info['dropped'] += 1
                continue

            # 2) 정밀 픽셀 단위 무효(음영/배경) 비율 체크
            inv = strip_invalid[:, rel_x:rel_x + ps]
            if inv.mean() >= args.max_invalid_ratio:
                info['dropped'] += 1
                continue

            raw_patch = strip[:, :, rel_x:rel_x + ps]
            # 3) 음영 영역이 특정 상수(예: 1, 10 등)로 균일하게 채워진 더미 패치인 경우 추가 제거 (표준편차 검사)
            if args.min_patch_std > 0:
                valid_pixels = raw_patch[:, ~inv]
                if valid_pixels.size == 0 or float(np.std(valid_pixels)) < args.min_patch_std:
                    info['dropped'] += 1
                    continue

            # contrast stretching: [0, p_upper] => [0, 1]
            patch = np.nan_to_num(raw_patch, nan=0., posinf=0., neginf=0.)
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
    parser.add_argument('--min_valid_val', type=_float_or_none, default=1e-5,
                        help="treat pixels whose bands are all <= this threshold as L1-D shadow/background padding ('none' to disable)")
    parser.add_argument('--min_patch_std', type=float, default=1e-4,
                        help='drop flat/constant shadow patches whose valid pixel std (before stretching) < this threshold (0 to disable)')
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
            msg = (f"skipped ({info['skipped']})" if 'skipped' in info else
                   f"valid_bbox={info['valid_bbox']}, saved={info['saved']}, dropped={info['dropped']}, "
                   f"p{args.percentile:g}={['%.4g' % v for v in info['p_upper']]}")
            print(f"[{i + 1}/{len(scenes)}] {info['scene']} ({info['width']}x{info['height']}): {msg}", flush=True)

    with open(os.path.join(args.dst_dir, 'make_tif_patches_info.json'), 'w') as f:
        json.dump(dict(args=vars(args), scenes=sorted(infos, key=lambda d: d['scene'])), f, indent=2)
    print(f"[make_tif_patches] saved {sum(d['saved'] for d in infos)} patches, dropped {sum(d['dropped'] for d in infos)} patches, "
          f'to {args.dst_dir} ({time.time() - start:.1f}s)')


def _float_or_none(v):
    return None if str(v).lower() in ('none', 'null', '') else float(v)


class _Serial:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def imap_unordered(self, fn, jobs): return map(fn, jobs)


if __name__ == '__main__':
    main()
