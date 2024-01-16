from mindcv.models import gmlp_Ti, gmlp_S, gmlp_B
import mindspore as ms


import numpy as np
input_np = np.random.rand(1, 3, 224, 224)
input_tensor = ms.Tensor(input_np, dtype=ms.float32)
net = gmlp_Ti()

total = sum([param.size for param in net.get_parameters()])
print("gmlp_Ti parameter:%fM" % (total/1e6)) 

# output = net(input_tensor)
# print(output.shape)

# net = gmlp_S()

# total = sum([param.size for param in net.get_parameters()])
# print("gmlp_S parameter:%fM" % (total/1e6)) 

# net = gmlp_B()

# total = sum([param.size for param in net.get_parameters()])
# print("gmlp_B parameter:%fM" % (total/1e6)) 