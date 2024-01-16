from mindspore import Tensor
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import numpy as np
from collections import OrderedDict as OD

device_id = 6
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=False)
context.set_context(device_id=device_id)

class A(nn.Cell):
    def __init__(self):
        super().__init__()
        conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        relu = nn.ReLU()
        od = OD()
        od['l1'] = conv
        od['l2'] = relu
        self.seq = nn.SequentialCell(od)

    def construct(self, x):
        x = self.seq(x)
        return x

x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
output = A()(x)
print(output)
