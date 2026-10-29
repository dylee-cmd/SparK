# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path, input_size, data_type='tif', tif_bands='', data_cache='', rebuild_data_cache=False) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :param data_type: 'tif': [0, 1] float tif patches made by `make_tif_patches.py`, used WITHOUT normalization;
                      'imagenet': the ImageNet folder (RGB images normalized by the ImageNet mean and std)
    :param tif_bands: (only for 'tif') 1-based band indices separated by commas, e.g., '1,2,3'; '': use all bands
    :param data_cache: (only for 'tif') path of the patch list cache; '': `<dataset_path>/.spark_cache/tif_patch_list.pt` (or under ~/.cache/spark)
    :param rebuild_data_cache: (only for 'tif') rescan the patch folder even if the cache is valid
    :return: the dataset used for pretraining
    """
    dataset_path = os.path.abspath(dataset_path)
    if data_type == 'tif':
        from utils.tif_dataset import TifPatchDataset, Clamp
        try:
            rrc = transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation, antialias=True)
        except TypeError:   # old torchvision has no `antialias`
            rrc = transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation)
        trans_train = transforms.Compose([  # the input is already a [0, 1] float tensor: no ToTensor, no Normalize
            rrc,
            transforms.RandomHorizontalFlip(),
            Clamp(0., 1.),
        ])
        bands = [int(b) for b in str(tif_bands).replace(' ', '').split(',') if b] or None
        import dist
        dataset_train = TifPatchDataset(
            folder=dataset_path, transform=trans_train, bands=bands, cache_path=data_cache, rebuild_cache=rebuild_data_cache,
            is_builder=dist.is_local_master(), barrier=dist.barrier,   # only the local rank 0 scans the folder; the others load its cache
        )
        print(f'[build_dataset_to_pretrain] {len(dataset_train)} tif patches found in {dataset_path}, bands={bands or "all"}')
        print_transform(trans_train, '[pre-train]')
        return dataset_train
    
    assert data_type == 'imagenet', f'unknown data_type: {data_type}'
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]
    
    dataset_train = ImageNetDataset(imagenet_folder=dataset_path, transform=trans_train, train=True)
    print_transform(trans_train, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
