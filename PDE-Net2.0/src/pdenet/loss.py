import mindspore as ms
import mindspore.numpy as ms_np


def huber_loss(param: ms.Tensor, threshold: float):
    upper_const = ms.Tensor([abs(threshold) + 1e-5], dtype=param.dtype)
    upper_param = ms.ops.concat((upper_const, param), axis=0)
    lower_const = ms.Tensor([abs(threshold) - 1e-5], dtype=param.dtype)
    lower_param = ms.ops.concat((lower_const, param), axis=0)

    upper_mask = ms.ops.abs(upper_param) > threshold
    lower_mask = ms.ops.abs(lower_param) <= threshold

    upper_loss = ms.ops.abs(upper_param) - 0.5 * threshold
    upper_loss = ms.ops.masked_select(upper_loss, upper_mask)
    lower_loss = ms.ops.pow(lower_param, 2) * 0.5 / threshold
    lower_loss = ms.ops.masked_select(lower_loss, lower_mask)

    loss = 0
    loss += ms_np.sum(upper_loss)
    loss += ms_np.sum(lower_loss)
    return loss


def moment_loss(model, threshold: float):
    r""" moment loss for regularization """
    all_param = [ms.Tensor([0.], dtype=ms.float32)]
    for param in model.fd_params():
        param = ms.ops.reshape(param, (-1, ))
        all_param.append(param)
    all_param = ms.numpy.concatenate(all_param, axis=0)
    loss = huber_loss(all_param, threshold)
    return loss


def symnet_loss(model, threshold):
    r""" symnet loss for regularization """
    all_param = []
    for param in model.symnet_params():
        param = ms.ops.reshape(param, (-1,))
        all_param.append(param)
    all_param = ms.numpy.concatenate(all_param, axis=0)
    loss = huber_loss(all_param, threshold)
    return loss


def data_loss(predict, label, dt: float):
    r"""
    MSE loss of (predict, label) at a certain step.
    :param predict: (batch_size, mesh_ndim=2, height, width)
    :param label: (batch_size, mesh_ndim=2, height, width)
    :param dt
    :return:
    """
    loss = ms.ops.mean(ms.ops.pow((predict - label) / dt, 2))
    return loss
