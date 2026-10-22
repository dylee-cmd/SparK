# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Check that the ResNet-vd defined in `models/resnetd.py` has exactly the same structure as the backbone of a
checkpoint (e.g., the RT-DETR R50-vd checkpoint of ai4rs, whose backbone is mmdet's `ResNetV1dPaddle`).

Usage (run under `pretrain/`):
    python check_resnetvd_ckpt.py --ckpt https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.pth
    python check_resnetvd_ckpt.py --ckpt /path/to/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.pth --arch resnet50vd
    # optionally, also export the backbone weights in the SparK (timm-style) format:
    python check_resnetvd_ckpt.py --ckpt /path/to/rtdetr.pth --export rtdetr_r50vd_backbone.pth

The weights saved by SparK pretraining (`resnet50vd_1kpretrained_timm_style.pth`) can be checked with
`--prefix ''`; they can be used as RT-DETR's `backbone.init_cfg=dict(type='Pretrained', checkpoint=...)` directly.
"""
import argparse
import sys

import torch
from timm import create_model

import models     # register the ResNet-vd models in timm


def load_backbone_state(ckpt: str, prefix: str, use_ema: bool):
    if ckpt.startswith(('http://', 'https://')):
        checkpoint = torch.hub.load_state_dict_from_url(ckpt, map_location='cpu', check_hash=False)
    else:
        try:
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)  # mmengine ckpts contain non-tensor meta info
        except TypeError:   # torch < 1.13 has no `weights_only`
            checkpoint = torch.load(ckpt, map_location='cpu')

    key = 'ema_state_dict' if use_ema else 'state_dict'
    if key in checkpoint:
        state = checkpoint[key]
    elif 'module' in checkpoint:    # SparK's `*_withdecoder_1kpretrained_spark_style.pth`
        state, prefix = checkpoint['module'], prefix or 'sparse_encoder.sp_cnn.'
    else:
        state = checkpoint
    print(f'[ckpt] top-level keys: {list(checkpoint.keys())[:10] if isinstance(checkpoint, dict) else type(checkpoint)}')

    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if prefix:
        others = sorted({k.split('.')[0] for k in state if not k.startswith(prefix)})
        print(f'[ckpt] ignored modules (not starting with {prefix!r}): {others}')
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='local path or url of the checkpoint')
    parser.add_argument('--arch', type=str, default='resnet50vd')
    parser.add_argument('--prefix', type=str, default='backbone.', help="prefix of the backbone keys in the checkpoint ('' for a backbone-only checkpoint)")
    parser.add_argument('--use_ema', action='store_true', help='check `ema_state_dict` instead of `state_dict`')
    parser.add_argument('--export', type=str, default='', help='save the backbone weights to this path (SparK/timm-style keys)')
    args = parser.parse_args()

    ckpt_state = load_backbone_state(args.ckpt, args.prefix, args.use_ema)
    model = create_model(args.arch, pretrained=False, num_classes=0, global_pool='')
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(ckpt_state))
    unexpected = sorted(set(ckpt_state) - set(model_state))
    mismatched = [(k, tuple(ckpt_state[k].shape), tuple(v.shape)) for k, v in model_state.items() if k in ckpt_state and ckpt_state[k].shape != v.shape]
    n_params_ckpt = sum(v.numel() for k, v in ckpt_state.items() if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
    n_params_model = sum(p.numel() for p in model.parameters())

    print(f'[check] #tensors: ckpt={len(ckpt_state)}, {args.arch}={len(model_state)}')
    print(f'[check] #params : ckpt={n_params_ckpt:,}, {args.arch}={n_params_model:,}')
    print(f'[check] missing keys    ({len(missing)}): {missing}')
    print(f'[check] unexpected keys ({len(unexpected)}): {unexpected}')
    print(f'[check] shape mismatches({len(mismatched)}): {mismatched}')

    ok = not missing and not unexpected and not mismatched
    if ok:
        model.load_state_dict(ckpt_state, strict=True)
        with torch.no_grad():
            feats = model.eval()(torch.rand(1, 3, 224, 224), hierarchical=True)
        print(f'[check] strict loading OK; feature maps: {[tuple(f.shape) for f in feats]}')
        print(f'[check] ==> the structure of `{args.arch}` is IDENTICAL to the checkpoint backbone.')
        if args.export:
            torch.save(ckpt_state, args.export)
            print(f'[export] backbone weights saved to {args.export}')
    else:
        print(f'[check] ==> the structure of `{args.arch}` is DIFFERENT from the checkpoint backbone.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
