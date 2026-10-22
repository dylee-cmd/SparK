# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
ResNet-vd (Baidu/PaddlePaddle ResNet-D variant) from "Bag of Tricks for Image Classification with Convolutional Neural Networks".

Compared with the timm ResNet-50 used in `pretrain/models/resnet.py`, ResNet-vd has:
    - a deep stem: three 3x3 convs (3->32->32->64) instead of a single 7x7 conv
    - the stride-2 of each bottleneck placed at the 3x3 conv (conv2)
    - an AvgPool(2x2) + 1x1 conv (stride 1) shortcut instead of a stride-2 1x1 conv shortcut

The module/parameter names are chosen to be **identical** to `ResNetV1dPaddle` used by RT-DETR in ai4rs/mmdet
(`projects/rtdetr/rtdetr/resnet.py`), so the `*_1kpretrained_timm_style.pth` saved by SparK can be loaded by
the RT-DETR backbone directly (no key conversion needed):
    stem.{0,1,3,4,6,7}.*                      (conv-bn-relu x 3)
    layer{i}.{j}.{conv1,bn1,conv2,bn2,conv3,bn3}.*
    layer{i}.0.downsample.{1,2}.*             (index 0 is the AvgPool, which has no params)
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.registry import register_model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path=0.):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(self.drop_path(out) + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path=0.):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)    # ResNet-vd (and ResNet-v1.5): stride at the 3x3 conv
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(self.drop_path(out) + identity)


class ResNetVd(nn.Module):
    def __init__(self, block, layers, num_classes=1000, global_pool='avg', drop_rate=0., drop_path_rate=0.,
                 stem_channels=64, zero_init_last_bn=True, **kwargs):
        super(ResNetVd, self).__init__()
        self.num_classes, self.drop_rate = num_classes, drop_rate

        # deep stem: 3x3(s2) -> 3x3 -> 3x3, each followed by BN and ReLU
        self.stem = nn.Sequential(
            conv3x3(3, stem_channels // 2, stride=2), nn.BatchNorm2d(stem_channels // 2), nn.ReLU(inplace=True),
            conv3x3(stem_channels // 2, stem_channels // 2), nn.BatchNorm2d(stem_channels // 2), nn.ReLU(inplace=True),
            conv3x3(stem_channels // 2, stem_channels), nn.BatchNorm2d(stem_channels), nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = stem_channels
        self.feature_map_channels = []
        num_blocks, block_idx = sum(layers), 0
        for stage_idx, (planes, n) in enumerate(zip((64, 128, 256, 512), layers)):
            dprs = [drop_path_rate * (block_idx + i) / (num_blocks - 1) for i in range(n)]   # stochastic depth linear decay rule (same as timm)
            block_idx += n
            self.add_module(f'layer{stage_idx + 1}', self._make_layer(block, planes, n, stride=1 if stage_idx == 0 else 2, dprs=dprs))
            self.feature_map_channels.append(planes * block.expansion)

        self.num_features = self.inplanes
        self.global_pool = nn.AdaptiveAvgPool2d(1) if global_pool else nn.Identity()
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def _make_layer(self, block, planes, num_blocks, stride, dprs):
        # the shortcut follows `ResLayer` of RT-DETR (ai4rs/projects/rtdetr/rtdetr/res_layer.py, `avg_down=True`)
        out_planes = planes * block.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                # for stride == 1 (the 1st stage of ResNet-50), mmdet/Paddle uses AvgPool2d(kernel_size=1), i.e., an identity
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False) if stride > 1 else nn.Identity(),
                conv1x1(self.inplanes, out_planes),
                nn.BatchNorm2d(out_planes),
            )
        else:   # the 1st stage of ResNet-18/34-vd
            downsample = nn.Sequential(conv1x1(self.inplanes, out_planes), nn.BatchNorm2d(out_planes))

        layers = [block(self.inplanes, planes, stride, downsample, drop_path=dprs[0])]
        self.inplanes = out_planes
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, drop_path=dprs[i]))
        return nn.Sequential(*layers)

    def get_downsample_ratio(self) -> int:
        return 32

    def get_feature_map_channels(self) -> List[int]:
        return self.feature_map_channels

    def forward(self, x, hierarchical=False):
        x = self.stem(x)
        x = self.maxpool(x)

        if hierarchical:
            ls = []
            x = self.layer1(x); ls.append(x)
            x = self.layer2(x); ls.append(x)
            x = self.layer3(x); ls.append(x)
            x = self.layer4(x); ls.append(x)
            return ls
        else:
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.flatten(self.global_pool(x))
            if self.drop_rate:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)
            x = self.fc(x)
            return x


@register_model
def resnet18vd(pretrained=False, **kwargs):
    return ResNetVd(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model
def resnet34vd(pretrained=False, **kwargs):
    return ResNetVd(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_model
def resnet50vd(pretrained=False, **kwargs):
    return ResNetVd(Bottleneck, [3, 4, 6, 3], **kwargs)


@register_model
def resnet101vd(pretrained=False, **kwargs):
    return ResNetVd(Bottleneck, [3, 4, 23, 3], **kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('resnet50vd')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())

    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()

    # check the forward function
    B, C, H, W = 4, 3, 224, 224
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])

    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio

    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
