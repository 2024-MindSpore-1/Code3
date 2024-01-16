import math

from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer

from mindcv.models.layers.weight_init import normal_


class DeepWiseConv2D(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, group=1, has_bias=False):
        super(DeepWiseConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.group = group
        self.has_bias = has_bias
        self.dilation = dilation
        self.data_format = "NCHW"
        self.bias_init = 'zeros'
        self.weight_init = 'normal'
        shape = out_channels * in_channels // group * kernel_size * kernel_size
        self.weight = Parameter(initializer(self.weight_init, shape), name='weight')
        self.bias_add = ops.BiasAdd(data_format=self.data_format)
        if self.has_bias:
            self.bias = Parameter(initializer(self.bias_init, [out_channels]), name='bias')
        fan_out = out_channels * kernel_size * kernel_size
        normal_(self.weight, std=math.sqrt(2 / fan_out))

    def construct(self, x):
        weight = self.weight
        weight = weight.reshape(self.out_channels, self.in_channels // self.group, self.kernel_size, self.kernel_size)
        weight = ops.cast(weight, x.dtype)
        output = ops.conv2d(x, weight, pad_mode="same", stride=self.stride, dilation=self.dilation, group=self.group)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output
