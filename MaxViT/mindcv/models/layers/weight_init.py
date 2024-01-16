import math

import numpy as np
from mindspore.common import initializer as weight_init
from mindspore.common.initializer import Initializer as MeInitializer


def assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    if nonlinearity == 'tanh':
        return 5.0 / 3
    if nonlinearity == 'relu':
        return math.sqrt(2.0)
    if nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))

    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_correct_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform(arr, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(arr, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, arr.shape)


def xavier_normal(arr, gain: float = 1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(arr)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return np.random.normal(0, std, arr.shape)


def _calculate_fan_in_and_fan_out(arr):
    """
    Calculate fan in and fan out
    """
    dimensions = len(arr.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for array with fewer than 2 dimensions")

    num_input_fmaps = arr.shape[1]
    num_output_fmaps = arr.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = arr[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def trunc_normal_tf(arr, mean=0., std=1., a=-2., b=2.):
    data = weight_init._init_truncated_normal(a, b, 0., 1., arr.shape)

    return data * std + mean


class KaimingUniform(MeInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingUniform, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        tmp = kaiming_uniform(arr, self.a, self.mode, self.nonlinearity)
        assignment(arr, tmp)


class XavierNormal(MeInitializer):
    def __init__(self, gain=1.):
        super(XavierNormal, self).__init__()
        self.gain = gain

    def _initialize(self, arr):
        tmp = xavier_normal(arr, self.gain)
        assignment(arr, tmp)


class TruncatedNormalTF(MeInitializer):
    def __init__(self, mean=0., std=1., a=-2., b=2.):
        super(TruncatedNormalTF, self).__init__()
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def _initialize(self, arr):
        tmp = trunc_normal_tf(arr, self.mean, self.std, self.a, self.b)
        assignment(arr, tmp)


def trunc_normal_(weight, std):
    weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=std),
                                            weight.shape,
                                            weight.dtype))


def trunc_normal_tf_(weight, std):
    weight.set_data(weight_init.initializer(TruncatedNormalTF(std=std),
                                            weight.shape,
                                            weight.dtype))


def normal_(weight, std):
    weight.set_data(weight_init.initializer(weight_init.Normal(sigma=std),
                                            weight.shape,
                                            weight.dtype))


def uniform_(weight, scale):
    weight.set_data(weight_init.initializer(weight_init.Uniform(scale=scale),
                                            weight.shape,
                                            weight.dtype))


def zeros_(weight):
    weight.set_data(weight_init.initializer(weight_init.Zero(),
                                            weight.shape,
                                            weight.dtype))


def ones_(weight):
    weight.set_data(weight_init.initializer(weight_init.One(),
                                            weight.shape,
                                            weight.dtype))


def constant_(weight, value):
    weight.set_data(weight_init.initializer(weight_init.Constant(value=value),
                                            weight.shape,
                                            weight.dtype))


def kaiming_uniform_(weight):
    weight.set_data(weight_init.initializer(KaimingUniform(),
                                            weight.shape,
                                            weight.dtype))


def xavier_uniform_(weight, gain=1):
    weight.set_data(weight_init.initializer(weight_init.XavierUniform(gain=gain),
                                            weight.shape,
                                            weight.dtype))


def xavier_normal_(weight, gain=1):
    weight.set_data(weight_init.initializer(XavierNormal(gain=gain),
                                            weight.shape,
                                            weight.dtype))


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.ndim > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    else:
        raise NotImplementedError
    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        normal_(tensor, std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        uniform_(bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
