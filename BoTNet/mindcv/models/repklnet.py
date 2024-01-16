# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import math

import numpy as np
from mindspore import nn, ops, Tensor

# from mindcv.models.dwconv import DeepWiseConv2D
from mindcv.models.layers import Dropout, DropPath
from mindcv.models.layers.weight_init import normal_, zeros_
from mindcv.models.registry import register_model

__all__ = ['replknet31_base', 'replknet_xlarge', 'replknet31_large', 'replknet31_base_7777']

BatchNorm2d = nn.BatchNorm2d


def DeepWiseConv2D(in_channels, out_channels, kernel_size, stride=1, group=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, group=group,
                     padding=kernel_size // 2, pad_mode='pad')


class LargeKernelReparam(nn.Cell):
    def __init__(self, channels, kernel, small_kernels=()):
        super(LargeKernelReparam, self).__init__()
        self.dw_large = nn.SequentialCell(
            DeepWiseConv2D(channels, channels, kernel_size=kernel, group=channels),
            BatchNorm2d(channels)
        )

        self.small_kernels = small_kernels
        small_kernels_cells = []
        for k in self.small_kernels:
            small_kernels_cells.append(nn.SequentialCell(
                DeepWiseConv2D(channels, channels, kernel_size=k, group=channels),
                BatchNorm2d(channels)))
        self.small_kernels_cells = nn.CellList(small_kernels_cells)

    def construct(self, inp):
        outp = self.dw_large(inp)
        for small_kernels_cell in self.small_kernels_cells:
            outp += small_kernels_cell(inp)
        return outp


class Mlp(nn.Cell):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0., ):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.fc1 = nn.SequentialCell(
            nn.Conv2d(in_channels, hidden_features, kernel_size=1),
            BatchNorm2d(hidden_features)
        )
        self.act = act_layer()
        self.fc2 = nn.SequentialCell(
            nn.Conv2d(hidden_features, out_features, kernel_size=1),
            BatchNorm2d(out_features)
        )
        self.drop1 = Dropout(drop)
        self.drop2 = Dropout(drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class RepLKBlock(nn.Cell):

    def __init__(self, channels, kernel, small_kernels=(), dw_ratio=1.0, mlp_ratio=4.0, drop_path=0.,
                 activation=nn.ReLU):
        super().__init__()

        self.pre_bn = BatchNorm2d(channels)
        self.pw1 = nn.SequentialCell(
            nn.Conv2d(channels, int(channels * dw_ratio), kernel_size=1),
            BatchNorm2d(int(channels * dw_ratio))
        )
        self.pw1_act = activation()
        self.dw = LargeKernelReparam(int(channels * dw_ratio), kernel, small_kernels=small_kernels)
        self.dw_act = activation()
        self.pw2 = nn.SequentialCell(
            nn.Conv2d(int(channels * dw_ratio), channels, 1),
            BatchNorm2d(channels)
        )

        self.pre_bn_2 = BatchNorm2d(channels)
        self.mlp = Mlp(in_channels=channels, hidden_channels=int(channels * mlp_ratio))

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        y = self.pre_bn(x)
        y = self.pw1_act(self.pw1(y))
        y = self.dw_act(self.dw(y))
        y = self.pw2(y)
        x = x + self.drop_path1(y)

        z = self.pre_bn_2(x)
        z = self.mlp(z)
        x = x + self.drop_path2(z)

        return x


class DownSample(nn.Cell):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            BatchNorm2d(out_channels),
            activation(),
            DeepWiseConv2D(out_channels, out_channels, kernel_size=3, stride=2, group=out_channels),
            BatchNorm2d(out_channels),
            activation()
        )

    def construct(self, x):
        return self.conv(x)


class Stem(nn.Cell):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.stem = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            BatchNorm2d(out_channels),
            activation(),
            DeepWiseConv2D(out_channels, out_channels, kernel_size=3, group=out_channels),
            BatchNorm2d(out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, 1),
            BatchNorm2d(out_channels),
            activation(),
            DeepWiseConv2D(out_channels, out_channels, 3, stride=2, group=out_channels),
            BatchNorm2d(out_channels),
            activation()
        )

    def construct(self, x):
        return self.stem(x)


class RepLKNet(nn.Cell):

    def __init__(
            self,
            in_channels=3,
            depths=(2, 2, 18, 2),
            dims=(128, 256, 512, 1024),
            kernel_sizes=(31, 29, 27, 13),
            small_kernels=(5,),
            dw_ratio=1.0,
            mlp_ratio=4.0,
            num_classes=1000,
            drop_path_rate=0.5,
            **kwargs
    ):
        super().__init__()

        self.stem = Stem(in_channels, dims[0])
        # stochastic depth
        dpr = (x for x in np.linspace(0, drop_path_rate, sum(depths)))  # stochastic depth decay rule

        blocks = []

        for stage, (depth, dim, ksize) in enumerate(zip(depths, dims, kernel_sizes)):
            for _ in range(depth):
                blocks.append(
                    RepLKBlock(dim, ksize, small_kernels=small_kernels,
                               dw_ratio=dw_ratio, mlp_ratio=mlp_ratio, drop_path=next(dpr))
                )
            if stage < len(depths) - 1:
                blocks.append(DownSample(dim, dims[stage + 1]))

        self.blocks = nn.CellList(blocks)
        self.norm = BatchNorm2d(dims[-1])
        self.head = nn.Dense(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights()

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = ops.ReduceMean(keep_dims=False)(x, (2, 3))
        return x

    def init_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                normal_(cell.weight, std=0.01)
                if cell.bias is not None:
                    zeros_(cell.bias)
            elif isinstance(cell, nn.Conv2d):
                # NOTE conv was left to pytorch default in my original init
                fan_out = cell.weight.shape[0] * cell.weight.shape[2] * cell.weight.shape[3]
                std = math.sqrt(2 / fan_out)
                normal_(cell.weight, std=std)
                if cell.bias is not None:
                    zeros_(cell.bias)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def replknet31_base(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(128, 256, 512, 1024), dw_ratio=1.0, in_channels=in_channels,
                    num_classes=num_classes, **kwargs)


@register_model
def replknet31_base_7777(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(128, 256, 512, 1024), dw_ratio=1.0, in_channels=in_channels,
                    num_classes=num_classes, kernel_sizes=(7, 7, 7, 7), **kwargs)


@register_model
def replknet31_large(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(192, 384, 768, 1536), dw_ratio=1.0, in_channels=in_channels,
                    num_classes=num_classes, **kwargs)


@register_model
def replknet_xlarge(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(256, 512, 1024, 2048), kernel_sizes=(27, 27, 27, 13), small_kernels=(), dw_ratio=1.5,
                    in_channels=in_channels, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    from mindspore import set_seed
    from mindspore import dtype as mstype

    set_seed(1234)
    model = replknet31_base()
    model.update_parameters_name(prefix='Ascend')
    data = Tensor(np.random.randn(1, 3, 224, 224), mstype.float32)
    out = model(data)
    print(out.shape)
    for name, param in model.parameters_and_names():
        print(name)
