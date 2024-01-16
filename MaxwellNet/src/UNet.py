""" 
Adapted and modified from https://github.com/jvanvugt/pytorch-unet

Modified parts:
Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

Original parts:
MIT License

Copyright (c) 2018 Joris

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import mindspore as ms
from mindspore import nn
import time


class UNet(nn.Cell):
    def __init__(self, in_channels=1, out_channels=2, depth=5, wf=6, norm='weight', up_mode='upconv'):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.down_path = nn.CellList()
        self.up_path = nn.CellList()

        prev_channels = int(in_channels)

        for i in range(depth):
            if i != depth - 1:
                if i == 0:
                    self.down_path.append(UNetConvBlock(
                        prev_channels, [wf * (2 ** i), wf * (2 ** i)], 3, 0, norm))
                else:
                    self.down_path.append(UNetConvBlock(
                        prev_channels, [wf * (2 ** i), wf * (2 ** i)], 3, 0, norm))
                prev_channels = int(wf * (2 ** i))
                self.down_path.append(nn.AvgPool2d(2, 2))
            else:
                self.down_path.append(UNetConvBlock(
                    prev_channels, [wf * (2 ** i), wf * (2 ** (i - 1))], 3, 0, norm))
                prev_channels = int(wf * (2 ** (i - 1)))

        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, [wf * (2 ** i), int(wf * (2 ** (i - 1)))], up_mode, 3, 0, norm))
            prev_channels = int(wf * (2 ** (i - 1)))

        self.last_conv = nn.Conv2d(
            prev_channels, out_channels, kernel_size=1, padding=0, has_bias=False, pad_mode='pad')

    def construct(self, scat_pot):
        blocks = []
        x = scat_pot
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0 and i != (len(self.down_path) - 1):
                blocks.append(x)
        for i, up in enumerate(self.up_path):

            x = up(x, blocks[-i - 1])

        x = self.last_conv(x)
        return x


class UNetConvBlock(nn.Cell):
    def __init__(self, in_size, out_size, kersize, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        if norm == 'weight':
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                                         padding=int(0), has_bias=True, pad_mode='pad'))
            block.append(nn.CELU())
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))

            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                                         padding=int(0), has_bias=True, pad_mode='pad'))
        elif norm == 'batch':
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                   padding=int(padding), has_bias=True, pad_mode='pad'))
            block.append(nn.BatchNorm2d(out_size[0]))
            block.append(nn.CELU())

            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                   padding=int(padding), has_bias=True, pad_mode='pad'))
            block.append(nn.BatchNorm2d(out_size[1]))

        elif norm == 'no':
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append((nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                    padding=int(0), has_bias=True, pad_mode='pad')))
            block.append(nn.CELU())
            block.append(nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="SYMMETRIC"))
            block.append((nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                    padding=int(0), has_bias=True, pad_mode='pad')))

        self.block = nn.SequentialCell(block)

    def construct(self, x):
        out = self.block(x)

        return out


class UNetUpBlock(nn.Cell):
    def __init__(self, in_size, out_size, up_mode, kersize, padding, norm):
        super(UNetUpBlock, self).__init__()
        block = []
        if up_mode == 'upconv':
            block.append(nn.Conv2dTranspose(in_size, in_size,
                         kernel_size=2, stride=2, has_bias=False))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2))
            block.append(nn.Conv2d(in_size, in_size,
                         kernel_size=1, has_bias=False))

        self.block = nn.SequentialCell(block)
        self.conv_block = UNetConvBlock(
            in_size * 2, out_size, kersize, padding, norm)

    def construct(self, x, bridge):
        up = self.block(x)
        out = ms.ops.concat([up, bridge], 1)
        out = self.conv_block(out)
        return out
