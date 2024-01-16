import numpy as np
from scipy.special import factorial
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.numpy import tensordot
from typing import Tuple, List, Union


r""" 
class _M2K is from mindflow.cell.neural_operators.m2k.py, 
if mindflow version is compatible with mindspore, just import:
    from mindflow.cell.neural_operators.m2k import _M2K
    (function k2m is added)
"""


class _M2K(nn.Cell):
    '''M2K module'''
    def __init__(self, shape, dtype=ms.float32):
        super(_M2K, self).__init__()
        self._shape = shape
        self._ndim = len(shape)
        self._m = []
        self._invm = []
        self.cast = ops.Cast()
        self.dtype = dtype
        for l in shape:
            zero_to_l = np.arange(l)
            mat = np.power(zero_to_l - l // 2, zero_to_l[:, None]) / factorial(zero_to_l[:, None])
            self._m.append(Tensor.from_numpy(mat))
            self._invm.append(Tensor.from_numpy(np.linalg.inv(mat)))

    @property
    def m(self):
        return self._m

    @property
    def invm(self):
        return self._invm

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    def _packdim(self, x):
        if x.ndim == self.ndim:
            x = x[None, :]
        x = x.view((-1, x.shape[1], x.shape[2]))
        return x

    def _apply_axis_left_dot(self, x, mats):
        x_shape = x.shape
        k = x.ndim - 1
        for i in range(k):
            x = tensordot(mats[k - i - 1].astype(self.dtype), x.astype(self.dtype), axes=[1, k])

        x = ops.transpose(x, (2, 0, 1))
        x = x.view(x_shape)
        return x

    def k2m(self, k):
        k_size = k.shape
        k = self.cast(k, self.dtype)
        k = self._packdim(k)
        m = self._apply_axis_left_dot(k, self.m)
        m = m.view(k_size)
        return m

    def construct(self, m):
        m_size = m.shape
        m = self.cast(m, self.dtype)
        m = self._packdim(m)
        k = self._apply_axis_left_dot(m, self.invm)
        k = k.view(m_size)
        return k


def periodic_pad_2d(inputs: ms.Tensor, pad_size: Tuple[int, int, int, int]):
    r"""
    pad inputs periodically.
    :param inputs: (batch_size, 2, height, width)
    :param pad_size: (left_size, right_size, top_size, bottom_size)
    :return:
    """
    left_pad = inputs[..., slice(-pad_size[0], None)]
    right_pad = inputs[..., slice(0, pad_size[1])]
    inputs = ms.ops.concat((left_pad, inputs, right_pad), axis=-1)
    top_pad = inputs[..., slice(-pad_size[2], None), :]
    bot_pad = inputs[..., slice(0, pad_size[3]), :]
    inputs = ms.ops.concat((top_pad, inputs, bot_pad), axis=-2)
    return inputs


def relative_l2_error(predict: ms.Tensor, label: ms.Tensor):
    r"""
    calculate the Relative Root Mean Square Error.
        math:
            error = \sqrt{\frac{\sum_{i=1}^{N}{(x_i-y_i)^2}}{sum_{i=1}^{N}{(y_i)^2}}}
    :param predict: (batch_size, mesh_ndim, height, width)
    :param label: (batch_size, mesh_ndim, height, width)
    :return: (batch_size, )
    """
    batch_size = predict.shape[0]
    predict = predict.reshape(batch_size, -1)
    label = label.reshape(batch_size, -1)
    diff_norms = ms.ops.square(predict - label).sum(axis=1)
    label_norms = ms.ops.square(label).sum(axis=1)
    rel_error = ms.ops.sqrt(diff_norms) / ms.ops.sqrt(label_norms)
    return rel_error
