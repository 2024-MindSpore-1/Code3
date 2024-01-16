from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import math
from mindspore import context

target = "Ascend"
device_id = 5

# init context
# context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
context.set_context(device_id=device_id)

weights = Tensor([-6.3884e-04, 3.8397e-03, -2.0675e-03, -1.8278e-02, 6.4828e-02,
                  -1.2870e-01, 1.8531e-01, 7.9200e-01, 1.8531e-01, -1.2870e-01,
                  6.4828e-02, -1.8278e-02, -2.0675e-03, 3.8397e-03, -6.3884e-04,
                  0.0000e+00])
num_channels = 1
conv_stride = 5
shape = (10, 1, 16012)
stdnormal = ops.StandardNormal(seed=2)
wave_to_conv = stdnormal(shape)
print("weights", weights)
print(type(weights))
# weight_ = ops.tile(weights, (num_channels, 1, 1))
weight_ = Tensor([[[-6.3884e-04, 3.8397e-03, -2.0675e-03, -1.8278e-02, 6.4828e-02,
                    -1.2870e-01, 1.8531e-01, 7.9200e-01, 1.8531e-01, -1.2870e-01,
                    6.4828e-02, -1.8278e-02, -2.0675e-03, 3.8397e-03, -6.3884e-04,
                    0.0000e+00]]])
in_channels = wave_to_conv.shape[1]
kernel_size = weight_.shape[2]
print("weight_.shape = ", weight_.shape)
# print("weight_ = ", weight_)
print("wave_to_conv.shape = ", wave_to_conv.shape)
print("num_channels = ", num_channels)
conv_wave = nn.Conv1d(
    in_channels, num_channels, kernel_size,
    # weight_init=np.tile(weights, (num_channels, 1, 1)),
    weight_init=weight_,
    stride=5,
    group=num_channels
)(wave_to_conv)
