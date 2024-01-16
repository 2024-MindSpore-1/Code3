from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as np
import mindspore.common.dtype as mstype
import math
from mindspore import context
target = "Ascend"
device_id = 2

# init context
# context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
context.set_context(device_id=device_id)

t_eq_zero_indices = Tensor([[False, False, False, False, False, False, False, False, False, False,
                             False, False, True, False, False, False, False, False, False, False,
                             False, False, False, False, False]])

t_not_eq_zero_indices = Tensor([[True, True, True, True, True, True, True, True, True, True,
                                 True, True, False, True, True, True, True, True, True, True,
                                 True, True, True, True, True]])

delta_t = Tensor([[-7.5000e-04, -6.8750e-04, -6.2500e-04, -5.6250e-04, -5.0000e-04,
                   -4.3750e-04, -3.7500e-04, -3.1250e-04, -2.5000e-04, -1.8750e-04,
                   -1.2500e-04, -6.2500e-05, 0.0000e+00, 6.2500e-05, 1.2500e-04,
                   1.8750e-04, 2.5000e-04, 3.1250e-04, 3.7500e-04, 4.3750e-04,
                   5.0000e-04, 5.6250e-04, 6.2500e-04, 6.8750e-04, 7.5000e-04]])

weights = Tensor([[2.4673e-04, 2.0964e-02, 7.3680e-02, 1.5487e-01, 2.5912e-01, 3.7946e-01,
                   5.0785e-01, 6.3572e-01, 7.5452e-01, 8.5632e-01, 9.3432e-01, 9.8330e-01,
                   1.0000e+00, 9.8330e-01, 9.3432e-01, 8.5632e-01, 7.5452e-01, 6.3572e-01,
                   5.0785e-01, 3.7946e-01, 2.5912e-01, 1.5487e-01, 7.3680e-02, 2.0964e-02,
                   2.4673e-04]])

# 需要将t_not_eq_zero_indices为true的索引组成Tensor
index = np.arange(0, t_not_eq_zero_indices.shape[1])
print(t_not_eq_zero_indices)
# t_not_eq_zero_indices = t_not_eq_zero_indices * index
# t_not_eq_zero_indices_ = ops.Cast()(t_not_eq_zero_indices, mstype.int32)
# t_eq_zero_indices_ = ops.Cast()(t_eq_zero_indices, mstype.int32)
t_not_eq_zero_indices_ = ops.MaskedSelect()(index, t_not_eq_zero_indices)
print(t_not_eq_zero_indices_)
print('delta_t[t_not_eq_zero_indices_]:', delta_t[0][t_not_eq_zero_indices_])
print("here")
lowpass_cutoff = 3960.0
sin = ops.Sin()
# weights_ = Tensor(weights)
weights_ = ops.masked_fill(weights, t_not_eq_zero_indices, 0)
weights[0][t_not_eq_zero_indices_] *= sin(
    2 * math.pi * lowpass_cutoff * delta_t[0][t_not_eq_zero_indices_]
) / (math.pi * delta_t[0][t_not_eq_zero_indices_])
print("weights_ = ", weights_)
print("weights = ", weights)
print("t_not_eq_zero_indices = ", t_not_eq_zero_indices)
print("t_eq_zero_indices = ", t_eq_zero_indices)
# output = ops.masked_fill(weights, t_not_eq_zero_indices, weights_)
# print(weights_ * t_eq_zero_indices_)
# print(weights * t_not_eq_zero_indices_)
print("weights = ", ops.masked_fill(weights, t_eq_zero_indices, 0))
# print("weights_ = ", ops.masked_fill(weights_, t_not_eq_zero_indices, 0))
weights = ops.masked_fill(weights, t_eq_zero_indices, 0) + weights_
# weights = weights * t_not_eq_zero_indices_ + weights_ * t_eq_zero_indices_
print("weights = ", weights)

# weights = ops.ZerosLike()(delta_t)
# inside_window_indices = Tensor([[True, True, True, True, True, True, True, True, True, True,
#                                  True, True, False, True, True, True, True, True, True, True,
#                                  True, True, True, True, True]])
# index = np.arange(0, inside_window_indices.shape[1])
# print("index = ", index)
# inside_window_indices = ops.MaskedSelect()(index, inside_window_indices)
# lowpass_cutoff = 3960.0
# print(weights)
# cos = ops.Cos()
# weights[inside_window_indices] = 0.5 * (
#             1
#             + cos(
#                 2
#                 * math.pi
#                 * lowpass_cutoff
#                 / lowpass_filter_width
#                 * delta_t[inside_window_indices]
#             )
#         )
# print(weights)


"""
# pytorch 版本
import torch
import math

t_eq_zero_indices = torch.tensor([[False, False, False, False, False, False, False, False, False, False,
                             False, False, True, False, False, False, False, False, False, False,
                             False, False, False, False, False]])

t_not_eq_zero_indices = torch.tensor([[True, True, True, True, True, True, True, True, True, True,
                                 True, True, False, True, True, True, True, True, True, True,
                                 True, True, True, True, True]])

delta_t = torch.tensor([[-7.5000e-04, -6.8750e-04, -6.2500e-04, -5.6250e-04, -5.0000e-04,
                   -4.3750e-04, -3.7500e-04, -3.1250e-04, -2.5000e-04, -1.8750e-04,
                   -1.2500e-04, -6.2500e-05, 0.0000e+00, 6.2500e-05, 1.2500e-04,
                   1.8750e-04, 2.5000e-04, 3.1250e-04, 3.7500e-04, 4.3750e-04,
                   5.0000e-04, 5.6250e-04, 6.2500e-04, 6.8750e-04, 7.5000e-04]])

weights = torch.tensor([[2.4673e-04, 2.0964e-02, 7.3680e-02, 1.5487e-01, 2.5912e-01, 3.7946e-01,
                   5.0785e-01, 6.3572e-01, 7.5452e-01, 8.5632e-01, 9.3432e-01, 9.8330e-01,
                   1.0000e+00, 9.8330e-01, 9.3432e-01, 8.5632e-01, 7.5452e-01, 6.3572e-01,
                   5.0785e-01, 3.7946e-01, 2.5912e-01, 1.5487e-01, 7.3680e-02, 2.0964e-02,
                   2.4673e-04]])
print("weights = ", weights)

# 支持bool类型
print(t_not_eq_zero_indices)

lowpass_cutoff = 3960.0
weights[t_not_eq_zero_indices] *= torch.sin(
    2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
) / (math.pi * delta_t[t_not_eq_zero_indices])
print("weights = ", weights)

# pytorch 运行结果
weights =  tensor([[2.4673e-04, 2.0964e-02, 7.3680e-02, 1.5487e-01, 2.5912e-01, 3.7946e-01,
         5.0785e-01, 6.3572e-01, 7.5452e-01, 8.5632e-01, 9.3432e-01, 9.8330e-01,
         1.0000e+00, 9.8330e-01, 9.3432e-01, 8.5632e-01, 7.5452e-01, 6.3572e-01,
         5.0785e-01, 3.7946e-01, 2.5912e-01, 1.5487e-01, 7.3680e-02, 2.0964e-02,
         2.4673e-04]])
tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True, False,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True]])
weights =  tensor([[-1.9622e-02, -9.5617e+00,  5.8702e+00,  8.6764e+01, -2.0675e+01,
         -2.7441e+02,  4.0568e+01,  6.4554e+02, -6.0322e+01, -1.4521e+03,
          7.4733e+01,  5.0073e+03,  1.0000e+00,  5.0073e+03,  7.4733e+01,
         -1.4521e+03, -6.0322e+01,  6.4554e+02,  4.0568e+01, -2.7441e+02,
         -2.0675e+01,  8.6764e+01,  5.8702e+00, -9.5617e+00, -1.9622e-02]])
"""
