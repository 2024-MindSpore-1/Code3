import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops

# from mindspore import context
# target = "Ascend"
# device_id = 4
#
# # init context
# # context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
# context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
# context.set_context(device_id=device_id)

x = Tensor(np.array([1, 2, 3, 4, 5]), mstype.int32)
all_new_value = Tensor(np.array([-1, -2, -3, -4, -5]), mstype.int32)
i = Tensor(np.array([True, False, True, True, False]), mstype.int32)
a = ops.Cast()(i, mstype.int32)
print(a)
mask = 1-a
y = x * mask + all_new_value * a
print(y)