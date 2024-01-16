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
input_x = Tensor([-6.3884e-04, 3.8397e-03, -2.0675e-03, -1.8278e-02, 6.4828e-02,
                  -1.2870e-01, 1.8531e-01, 7.9200e-01, 1.8531e-01, -1.2870e-01,
                  6.4828e-02, -1.8278e-02, -2.0675e-03, 3.8397e-03, -6.3884e-04,
                  0.0000e+00], mindspore.float32)
# input_x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
print("output_tile")
multiples = (2, 1, 1)
output_tile = np.tile(input_x, multiples)
print(output_tile.shape)
print(output_tile)
print("output_broadcast_to")
output_broadcast_to = ops.broadcast_to(input_x, (2, 1, -1))
print(output_broadcast_to.shape)
print(output_broadcast_to)
