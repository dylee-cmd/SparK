# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import math
import os
import sys
import time
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import misc, lamb
from utils.imagenet import build_dataset_to_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'boolean value expected, got {v!r}')


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('SparK pretraining', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # environment
    parser.add_argument('--exp_name', type=str, default='your_exp_name', help='experiment name')
    parser.add_argument('--exp_dir', type=str, default='your_exp_dir', help='will be created if not exists')
    parser.add_argument('--data_path', type=str, default='imagenet_data_path', help='for --data_type=tif: the --dst_dir of make_tif_patches.py')
    parser.add_argument('--data_type', type=str, default='tif', choices=['tif', 'imagenet'],
                        help="'tif': [0, 1] float tif patches made by make_tif_patches.py (no normalization); 'imagenet': the ImageNet folder")
    parser.add_argument('--tif_bands', type=str, default='',
                        help="(only for --data_type=tif) 1-based band indices to use, e.g., '1,2,3' or '3,2,1'; '': all bands (each patch must have 1 or 3 bands)")
    parser.add_argument('--init_weight', type=str, default='', help='use some checkpoint as model weight initialization; ONLY load model weights')
    parser.add_argument('--resume_from', type=str, default='', help='resume the experiment from some checkpoint.pth; load model weights, optimizer states, and last epoch')
    parser.add_argument('--tb_lg_dir', type=str, default='', help="tensorboard log directory ('': <exp_dir>/tensorboard_log)")
    
    # SparK hyperparameters
    parser.add_argument('--mask', type=float, default=0.6, help='mask ratio, should be in (0, 1)')
    
    # encoder hyperparameters
    parser.add_argument('--model', type=str, default='resnet50', help="see the keys of 'pretrain_default_model_kwargs' in models/__init__.py")
    parser.add_argument('--input_size', type=int, default=224, help='input resolution, a multiple of 32')
    parser.add_argument('--sbn', type=str2bool, default=True, help='use SyncBatchNorm (only when DDP is initialized), e.g., --sbn=False')
    
    # data hyperparameters
    parser.add_argument('--bs', type=int, default=4096, help='the TOTAL batch size of all GPUs')
    parser.add_argument('--dataloader_workers', type=int, default=8, help='dataloader workers of each GPU')
    
    # pre-training hyperparameters
    parser.add_argument('--dp', type=float, default=0.0, help='drop path rate; 0: use the default of the model in models/__init__.py')
    parser.add_argument('--base_lr', type=float, default=2e-4, help='the actual lr = base_lr * bs / 256')
    parser.add_argument('--wd', type=float, default=0.04, help='weight decay at the beginning')
    parser.add_argument('--wde', type=float, default=0.2, help='weight decay at the end')
    parser.add_argument('--ep', type=int, default=1600, help='pretraining epochs')
    parser.add_argument('--wp_ep', type=int, default=40, help='warmup epochs')
    parser.add_argument('--clip', type=float, default=5., help='gradient clipping (<= 0: no clipping)')
    parser.add_argument('--opt', type=str.lower, default='lamb', choices=['lamb', 'adamw'], help='optimizer')
    parser.add_argument('--ada', type=float, default=0., help='beta2 of the optimizer; 0: 0.95 for ResNets, 0.999 for ConvNeXts')
    return parser


def init_dist_and_get_args() -> argparse.Namespace:
    args = get_args_parser().parse_args()
    args.cmd = ' '.join(sys.argv[1:])
    args.commit_id = os.popen('git rev-parse HEAD').read().strip() or '[unknown]'
    args.commit_msg = (os.popen('git log -1').read().strip().splitlines() or ['[unknown]'])[-1].strip()
    
    # the experiment directory (only letters, digits and '-' are kept in its name)
    e = os.path.abspath(args.exp_dir)
    d, e = os.path.dirname(e), os.path.basename(e)
    e = ''.join(ch if (ch.isalnum() or ch == '-') else '_' for ch in e)
    args.exp_dir = os.path.join(d, e)
    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_txt_name = os.path.join(args.exp_dir, 'pretrain_log.txt')
    args.tb_lg_dir = args.tb_lg_dir or os.path.join(args.exp_dir, 'tensorboard_log')
    try:
        os.makedirs(args.tb_lg_dir, exist_ok=True)
    except:
        pass
    
    misc.init_distributed_environ(exp_dir=args.exp_dir)
    
    # runtime args (updated automatically)
    if not dist.initialized():
        args.sbn = False
    args.device = dist.get_device()
    args.batch_size_per_gpu = args.bs // dist.get_world_size()
    args.glb_batch_size = args.batch_size_per_gpu * dist.get_world_size()
    args.densify_norm = ''
    if 'resnet' in args.model:
        args.ada = args.ada or 0.95
        args.densify_norm = 'bn'
    if 'convnext' in args.model or 'cnx' in args.model:
        args.ada = args.ada or 0.999
        args.densify_norm = 'ln'
    args.lr = args.base_lr * args.glb_batch_size / 256
    args.wde = args.wde or args.wd
    
    # for logging
    args.first_logging = True
    args.last_loss, args.cur_ep, args.remain_time, args.finish_time = 0., '', '', ''
    return args


def log_epoch(args: argparse.Namespace):
    if not dist.is_local_master():
        return
    
    if args.first_logging:
        args.first_logging = False
        with open(args.log_txt_name, 'w') as fp:
            json.dump({
                'name': args.exp_name, 'cmd': args.cmd, 'git_commit_id': args.commit_id, 'git_commit_msg': args.commit_msg,
                'model': args.model,
            }, fp)
            fp.write('\n\n')
    
    with open(args.log_txt_name, 'a') as fp:
        json.dump({
            'cur_ep': args.cur_ep,
            'last_L': args.last_loss,
            'rema': args.remain_time, 'fini': args.finish_time,
        }, fp)
        fp.write('\n')


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_pt():
    args = init_dist_and_get_args()
    print(f'initial args:\n' + '\n'.join(f'  {k}: {v}' for k, v in vars(args).items()))
    log_epoch(args)
    
    # build data
    print(f'[build data for pre-training] ...\n')
    dataset_train = build_dataset_to_pretrain(args.data_path, args.input_size, args.data_type, args.tif_bands)
    data_loader_train = DataLoader(
        dataset=dataset_train, num_workers=args.dataloader_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn
    )
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    # build encoder and decoder
    enc: encoder.SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False)
    dec = LightDecoder(enc.downsample_raito, sbn=args.sbn)
    model_without_ddp = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask,
        densify_norm=args.densify_norm, sbn=args.sbn,
    ).to(args.device)
    print(f'[PT model] model = {model_without_ddp}\n')
    
    # the model has been randomly initialized in their construction time
    # now try to load some checkpoint as model weight initialization; this ONLY loads the model weights
    misc.initialize_weight(args.init_weight, model_without_ddp)
    
    if dist.initialized():
        model: DistributedDataParallel = DistributedDataParallel(model_without_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    else:
        model = LocalDDP(model_without_ddp)
    
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume the experiment from some checkpoint.pth; this will load model weights, optimizer states, and last epoch (ep_start)
    # if loaded, ep_start will be greater than 0
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model_without_ddp, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Recon Loss: {performance_desc}')
    else:   # perform pre-training
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        min_loss = 1e9
        print(f'[PT start] from ep{ep_start}')
        
        pt_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = pre_train_one_ep(ep, args, tb_lg, itrt_train, iters_train, model, optimizer)
            last_loss = stats['last_loss']
            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f}'
            misc.save_checkpoint_with_meta_info_and_opt_state(f'{args.model}_withdecoder_1kpretrained_spark_style.pth', args, ep, performance_desc, model_without_ddp.state_dict(with_config=True), optimizer.state_dict())
            misc.save_checkpoint_model_weights_only(f'{args.model}_1kpretrained_timm_style.pth', args, model_without_ddp.sparse_encoder.sp_cnn.state_dict())
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Min/Last Recon Loss: {performance_desc},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            log_epoch(args)
            
            tb_lg.update(min_loss=min_loss, head='train', step=ep)
            tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            tb_lg.flush()
        
        # finish pre-training
        tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Recon Loss: {performance_desc},    Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        tb_lg.close()
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    log_epoch(args)


def pre_train_one_ep(ep, args: argparse.Namespace, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model: DistributedDataParallel, optimizer):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch {ep}:'
    
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, inp in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward and backward
        inp = inp.to(args.device, non_blocking=True)
        SparK.forward
        loss = model(inp, active_b1ff=None, vis=False)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        
        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        torch.cuda.synchronize()
        
        # log
        me.update(last_loss=loss)
        me.update(max_lr=max_lr)
        tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head='train_hp')
        tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


if __name__ == '__main__':
    main_pt()
