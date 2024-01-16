# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
from functools import partial

from mindspore import nn

if os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d


class ConvBlockSingleInput(nn.Cell):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, group=1,
                 norm_layer=partial(BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None, return_x_2=True):
        super(ConvBlockSingleInput, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, group=group)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.return_x_2 = return_x_2

    def construct(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)
        x += residual
        x = self.act3(x)

        if self.return_x_2:
            return x, x2
        else:
            return x


class ConvBlockDoubleInput(nn.Cell):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, group=1,
                 norm_layer=partial(BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None, return_x_2=True):
        super(ConvBlockDoubleInput, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, group=group)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.return_x_2 = return_x_2

    def construct(self, x, x_t):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if self.return_x_2:
            return x, x2
        else:
            return x
