import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import context

target = "Ascend"
device_id = 5

# init context
# context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
context.set_context(device_id=device_id)
uniform_real = ops.UniformReal(seed=2)
drop_prob = 1.0
print("start")
print(uniform_real((1,))[0])
if int(uniform_real((1,))[0]) > drop_prob:
    print("true")
else:
    print("false")
print("end")