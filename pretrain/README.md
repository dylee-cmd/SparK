## Preparation for ImageNet-1k pretraining

See [/INSTALL.md](/INSTALL.md) to prepare `pip` dependencies and the ImageNet dataset.

**Note: for neural network definitions, we directly use `timm.models.ResNet` and [official ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py).**


## Tutorial for pretraining your own CNN model

See [/pretrain/models/custom.py](/pretrain/models/custom.py). Your todo list is:

- implement `get_downsample_ratio` in [/pretrain/models/custom.py line20](/pretrain/models/custom.py#L20).
- implement `get_feature_map_channels` in [/pretrain/models/custom.py line29](/pretrain/models/custom.py#L29).
- implement `forward` in [/pretrain/models/custom.py line38](/pretrain/models/custom.py#L38).
- define `your_convnet(...)` with `@register_model` in [/pretrain/models/custom.py line54](/pretrain/models/custom.py#L53-L54).
- add default kwargs of `your_convnet(...)` in [/pretrain/models/\_\_init\_\_.py line34](/pretrain/models/__init__.py#L34).
- **Note: see [#54](/../../issues/54) if your CNN contains SE module or global average pooling layer, and see [#56](/../../issues/56) if it contains GroupNorm**.

Then run the experiment with `--model=your_convnet`.


## Pretraining ResNet-vd (Baidu/PaddlePaddle ResNet-D, e.g., the backbone of RT-DETR)

[/pretrain/models/resnetd.py](/pretrain/models/resnetd.py) provides `resnet18vd`, `resnet34vd`, `resnet50vd` and `resnet101vd`, whose parameter names are identical to `ResNetV1dPaddle` used by RT-DETR (mmdet/ai4rs).
Run the experiment with `--model=resnet50vd`; the saved `resnet50vd_1kpretrained_timm_style.pth` can be used directly as RT-DETR's `backbone.init_cfg=dict(type='Pretrained', checkpoint=...)`.

To check that the structure is the same as the backbone of an existing checkpoint:
```shell script
cd /path/to/SparK/pretrain
python3 check_resnetvd_ckpt.py --arch=resnet50vd --ckpt=https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.pth
```


## Pretraining on remote sensing scenes (float [0, 1] tif patches)

1. Crop the large scenes into patches with [/pretrain/make_tif_patches.py](/pretrain/make_tif_patches.py) (requires GDAL: `conda install -c conda-forge gdal`).
   Each scene is contrast-stretched by its 99th percentile (per band) to [0, 1] and saved as **16-bit float** GeoTIFF patches (not converted to uint8;
   half the size of float32, error <= 4.9e-4, read as float32 for training). Use `--dtype=float32` for full precision.
   Patches whose no-data/shadow pixels (NaN, or all bands == nodata value, 0 by default) are >= 20% are dropped.
```shell script
cd /path/to/SparK/pretrain
python3 make_tif_patches.py --src_dir=/path/to/scenes --dst_dir=/path/to/patches --patch_size=224 --stride=100 --workers=8
# options: --percentile=99 --percentile_mode=band|global --max_invalid_ratio=0.2 --nodata=0 --bands 3 2 1 --dtype=float16|float32 --compress=DEFLATE
```
2. Pretrain with `--data_type=tif` (the default), giving the `--dst_dir` of step 1 as `--data_path` (patches are searched recursively).
   The patches are read as they are (no ImageNet normalization); 1-band patches are repeated to 3 channels.
   If the patches have other than 1 or 3 bands (e.g., RGB+NIR), select them by `--tif_bands=1,2,3` (1-based); this is checked before training starts.
```shell script
# DDP on 8 GPUs of 1 machine (--bs is the total batch size of all GPUs, and must be divisible by the number of GPUs)
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 main.py \
  --data_path=/path/to/patches --data_type=tif --exp_name=rs_r101vd --exp_dir=/path/to/exp \
  --model=resnet101vd --input_size=224 --bs=512
# for 4-band patches: add `--tif_bands=1,2,3`; to resume: add `--resume_from=/path/to/exp/resnet101vd_withdecoder_1kpretrained_spark_style.pth`
# for bfloat16 mixed precision (A100/H100/RTX30xx or newer): add `--bf16`
```
**Patch list cache**: scanning millions of patch files is slow, so the sorted patch list is cached at `<data_path>/.spark_cache/tif_patch_list.pt`
(or under `~/.cache/spark/` if `--data_path` is not writable; set another path by `--data_cache`). Only the local rank 0 scans the folder; the other ranks load the cache.
The cache is rebuilt automatically when a scene folder (`<data_path>/<scene>/`) changes; use `--rebuild_data_cache` to force a rescan (e.g., after changing deeper sub-folders).

**Note for a large patch set**: the defaults `--ep=1600 --wp_ep=40` are for ImageNet (1.28M images, 312 iterations/epoch with bs=4096).
`make_tif_patches.py --stride=100` with 224 patches gives ~5x overlapping patches, so one epoch can have >100k iterations (see `iters_train` in the log).
Choose `--ep` and `--wp_ep` (can be a fraction) by the total number of iterations, e.g., for ~500k iterations in total with 130k iterations/epoch: `--ep=4 --wp_ep=0.1`.
Use `--data_type=imagenet` for the original ImageNet pipeline.

Or use [/pretrain/train_resnet101_tif.sh](/pretrain/train_resnet101_tif.sh), which launches DDP with `torchrun` (all visible GPUs by default) and resumes automatically:
```shell script
bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir                                  # 1 node, all GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_PER_GPU=32 bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir
NNODES=2 NODE_RANK=0 MASTER_ADDR=<master_ip> bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir   # on each node
```


## Tutorial for pretraining on your own dataset

See the comment of `build_dataset_to_pretrain` in [line55 of /pretrain/utils/imagenet.py](/pretrain/utils/imagenet.py#L55). Your todo list:

- Define a subclass of `torch.utils.data.Dataset` for your own unlabeled dataset, to replace our `ImageNetDataset`.
- Use `args.data_path` and `args.input_size` to help build your dataset, with `--data_path=... --input_size=...` to specify them.
- Note the batch size `--bs` is the total batch size of all GPU, which may need to be adjusted based on your dataset size. FYI: we use `--bs=4096` for ImageNet, which contains 1.28 million images.

**If your dataset is relatively small**, you can try `--init_weight=/path/to/res50_withdecoder_1kpretrained_spark_style.pth` to do your pretraining *from our pretrained weights*, rather than *form scratch*.

## Debug on 1 GPU (without DistributedDataParallel)

Use a small batch size `--bs=32` for avoiding OOM.

```shell script
python3 main.py --exp_name=debug --data_path=/path/to/imagenet --model=resnet50 --bs=32
```


## Pretraining Any Model on ImageNet-1k (224x224)

For pretraining, run [/pretrain/main.py](/pretrain/main.py) with `torchrun`.
**It is required to specify** the ImageNet data folder (`--data_path`), your experiment name & log dir (`--exp_name` and `--exp_dir`, automatically created if not exists), and the model name (`--model`, valid choices see the keys of 'pretrain_default_model_kwargs' in [/pretrain/models/\_\_init\_\_.py line34](/pretrain/models/__init__.py#L34)).

We use the **same** pretraining configurations (lr, batch size, etc.) for all models (ResNets and ConvNeXts) in 224 pretraining.
Their **names** and **default values** are in `get_args_parser()` of [/pretrain/main.py](/pretrain/main.py) (or run `python3 main.py --help`).
All these default configurations (like batch size 4096) would be used, unless you specify some like `--bs=512`.

**Note: the batch size `--bs` is the total batch size of all GPU, and the learning rate `--base_lr` is the base lr. The actual lr would be `lr = base_lr * bs / 256`, as in `init_dist_and_get_args()` of [/pretrain/main.py](/pretrain/main.py). So do not use `--lr` to specify a lr (that will be ignored)**

Here is an example to pretrain a ResNet50 on an 8-GPU single machine (we use DistributedDataParallel), overwriting the default batch size to 512:
```shell script
$ cd /path/to/SparK/pretrain
$ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=<some_port> main.py \
  --data_path=/path/to/imagenet --exp_name=<your_exp_name> --exp_dir=/path/to/logdir \
  --model=resnet50 --bs=512
```

For multiple machines, change the `--nnodes`, `--node_rank`, `--master_address` and `--master_port` to your configurations. E.g.:
```shell script
$ torchrun --nproc_per_node=8 --nnodes=<your_nnodes> --node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port> main.py \
  ...
```

## Pretraining ConvNeXt-Large on ImageNet-1k (384x384)

For 384 pretraining we use a larger mask ratio (0.75), a half batch size (2048), and a double base learning rate (4e-4):

```shell script
$ cd /path/to/SparK/pretrain
$ torchrun --nproc_per_node=8 --nnodes=<your_nnodes> --node_rank=<rank_starts_from_0> --master_address=<some_address> --master_port=<some_port> main.py \
  --data_path=/path/to/imagenet --exp_name=<your_exp_name> --exp_dir=/path/to/logdir \
  --model=convnext_large --input_size=384 --mask=0.75 --bs=2048 --base_lr=4e-4
```

## Logging

See files in your `--exp_dir` to track your experiment:

- `<model>_withdecoder_1kpretrained_spark_style.pth`: saves model and optimizer states, current epoch, current reconstruction loss, etc.; can be used to resume pretraining; can also be used for visualization in [/pretrain/viz_reconstruction.ipynb](/pretrain/viz_reconstruction.ipynb)
- `<model>_1kpretrained_timm_style.pth`: can be used for downstream finetuning
- `pretrain_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: the command of this experiment
    
    It also reports the loss and remaining pretraining time.

- `tensorboard_log/`: saves a lot of tensorboard logs including loss values, learning rates, gradient norms and more things. Use `tensorboard --logdir /path/to/this/tensorboard_log/ --port 23333` for viz.
- `stdout_backup.txt` and `stderr_backup.txt`: backups stdout/stderr.

## Resuming

Specify `--resume_from=path/to/<model>_withdecoder_1kpretrained_spark_style.pth` to resume pretraining. Note this is different from `--init_weight`:

- `--resume_from` will load three things: model weights, optimizer states, and current epoch, so it is used to resume some interrupted experiment (will start from that 'current epoch').
- `--init_weight` ONLY loads the model weights, so it's just like a model initialization (will start from epoch 0).


## Regarding sparse convolution

We do not use sparse convolutions in this pytorch implementation, due to their limited optimization on modern hardware.
As can be found in [/pretrain/encoder.py](/pretrain/encoder.py), we use masked dense convolution to simulate submanifold sparse convolution.
We also define some sparse pooling or normalization layers in [/pretrain/encoder.py](/pretrain/encoder.py).
All these "sparse" layers are implemented through pytorch built-in operators.


## Some details: how we mask images and how to set the patch size

In SparK, the mask patch size **equals to** the downsample ratio of the CNN model (so there is no configuration like `--patch_size=32`).

Here is the reason: when we do mask, we:

1. first generate the binary mask for the **smallest** resolution feature map, i.e., generate the `_cur_active` or `active_b1ff` in [/pretrain/spark.py line86-87](/pretrain/spark.py#L86-L87), which is a `torch.BoolTensor` shaped as `[B, 1, fmap_h, fmap_w]`, and would be used to mask the smallest feature map.
3. then progressively upsample it (i.e., expand its 2nd and 3rd dimensions by calling `repeat_interleave(..., dim=2)` and `repeat_interleave(..., dim=3)` in [/pretrain/encoder.py line16](/pretrain/encoder.py#L16)), to mask those feature maps ([`x` in line21](/pretrain/encoder.py#L21)) with larger resolutions .

So if you want a patch size of 16 or 8, you should actually define a new CNN model with a downsample ratio of 16 or 8.
See [Tutorial for pretraining your own CNN model (above)](https://github.com/keyu-tian/SparK/tree/main/pretrain/#tutorial-for-pretraining-your-own-cnn-model).
