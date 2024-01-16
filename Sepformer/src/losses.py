from itertools import permutations
import mindspore
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds


class PitWrapper(nn.Cell):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.

    Permutation invariance is calculated over the sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        Torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    # def __init__(self, base_loss):
    def __init__(self):
        super(PitWrapper, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.stack = ops.Stack()
        # self.base_loss = base_loss

        self.sum = ops.ReduceSum(keep_dims=True)
        self.sum_ = ops.ReduceSum()

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat : torch.Tensor
            Tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current batch, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        # shape = range(loss_mat.shape[0])
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[tuple(range(loss_mat.shape[0])), p].mean()
            # c_loss = loss_mat[shape, p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """
        Arguments
        ---------
        pred : torch.Tensor
            Network prediction for the current example, tensor of
            shape [..., sources].
        target : torch.Tensor
            Target for the current example, tensor of shape [..., sources].

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current example, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.shape[-1]
        # pred = pred.unsqueeze(-2).repeat(
        #     *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        # )
        # expand_dims = ops.ExpandDims()
        # pred = np.tile(self.expand_dims(pred, -2), (*[1 for x in range(len(pred.shape) - 1)], n_sources, 1))
        pred = np.tile(self.expand_dims(pred, -2), (n_sources, 1))

        # target = target.unsqueeze(-1).repeat(
        #     1, *[1 for x in range(len(target.shape) - 1)], n_sources
        # )
        # target = np.tile(self.expand_dims(target, -1), (1, *[1 for x in range(len(target.shape) - 1)], n_sources))
        target = np.tile(self.expand_dims(target, -1), (1, n_sources))

        # loss_mat = self.base_loss(pred, target)
        loss_mat = cal_si_snr(pred, target)
        # assert (
        #     len(loss_mat.shape) >= 2
        # ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(axis=mean_over[:-2])

        return self._fast_pit(loss_mat)

    # def _cal_si_snr(self, source, estimate_source):
    #     """Calculate SI-SNR.
    #
    #     Arguments:
    #     ---------
    #     source: [T, B, C],
    #         Where B is batch size, T is the length of the sources, C is the number of sources
    #         the ordering is made so that this loss is compatible with the class PitWrapper.
    #
    #     estimate_source: [T, B, C]
    #         The estimated source.
    #
    #     Example:
    #     ---------
    #     >>> import numpy as np
    #     >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    #     >>> xhat = x[:, (1, 0)]
    #     >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    #     >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    #     >>> si_snr = -cal_si_snr(x, xhat)
    #     >>> print(si_snr)
    #     tensor([[[ 25.2142, 144.1789],
    #              [130.9283,  25.2142]]])
    #     """
    #     # sum = ops.ReduceSum(keep_dims=True)
    #     EPS = 1e-8
    #     # assert source.shape() == estimate_source.shape()
    #     # device = estimate_source.device.type
    #
    #     # source_lengths = torch.tensor(
    #     #     [estimate_source.shape[0]] * estimate_source.shape[1], device=device
    #     # )
    #
    #     # source_lengths = Tensor([estimate_source.shape[0]] * estimate_source.shape[1])
    #     source_lengths = [estimate_source.shape[0]] * estimate_source.shape[1]
    #     source_lengths = Tensor(source_lengths, dtype=mindspore.int32)
    #     mask = self._get_mask(source, source_lengths)
    #     estimate_source *= mask
    #
    #     # num_samples = (
    #     #     source_lengths.contiguous().reshape(1, -1, 1).float()
    #     # )  # [1, B, 1]
    #     num_samples = (
    #         source_lengths.copy().reshape((1, -1, 1)).astype("float32")
    #     )  # [1, B, 1]
    #     mean_target = self.sum(source, 0) / num_samples
    #     mean_estimate = (
    #             self.sum(estimate_source, 0) / num_samples
    #     )
    #     zero_mean_target = source - mean_target
    #     zero_mean_estimate = estimate_source - mean_estimate
    #     # mask padding position along T
    #     zero_mean_target *= mask
    #     zero_mean_estimate *= mask
    #
    #     # Step 2. SI-SNR with PIT
    #     # reshape to use broadcast
    #     s_target = zero_mean_target  # [T, B, C]
    #     s_estimate = zero_mean_estimate  # [T, B, C]
    #     # s_target = <s', s>s / ||s||^2
    #     dot = self.sum(s_estimate * s_target, 0)  # [1, B, C]
    #     s_target_energy = (self.sum(s_target ** 2, 0) + EPS)  # [1, B, C]
    #     proj = dot * s_target / s_target_energy  # [T, B, C]
    #     # e_noise = s' - s_target
    #     e_noise = s_estimate - proj  # [T, B, C]
    #     # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    #     # sum_ = ops.ReduceSum()
    #     si_snr_beforelog = self.sum_(proj ** 2, 0) / (
    #             self.sum_(e_noise ** 2, 0) + EPS
    #     )
    #     si_snr = 10 * np.log10(si_snr_beforelog + EPS)  # [B, C]
    #
    #     # expand_dims = ops.ExpandDims()
    #     si_snr = self.expand_dims(si_snr, 0)
    #     # return -ops.ExpandDims()(si_snr, 0)
    #     return -si_snr
    #
    # def _get_mask(self, source, source_lengths):
    #     """
    #     Arguments
    #     ---------
    #     source : [T, B, C]
    #     source_lengths : [B]
    #
    #     Returns
    #     -------
    #     mask : [T, B, 1]
    #
    #     Example:
    #     ---------
    #     >>> source = torch.randn(4, 3, 2)
    #     >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    #     >>> mask = get_mask(source, source_lengths)
    #     >>> print(mask)
    #     tensor([[[1.],
    #              [1.],
    #              [1.]],
    #     <BLANKLINE>
    #             [[1.],
    #              [0.],
    #              [1.]],
    #     <BLANKLINE>
    #             [[0.],
    #              [0.],
    #              [1.]],
    #     <BLANKLINE>
    #             [[0.],
    #              [0.],
    #              [1.]]])
    #     """
    #     T, B, _ = source.shape
    #     # mask = source.new_ones((T, B, 1))
    #     mask = Tensor(np.ones((T, B, 1)), dtype=source.dtype)
    #     for i in range(B):
    #         mask[source_lengths[i]:, i, :] = 0
    #     return mask

    def construct(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss : torch.Tensor
                Permutation invariant loss for current examples, tensor of
                shape [batch]

            perms : list
                List of indexes for optimal permutation of the inputs over
                sources.
                e.g., [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            label = targets[i]
            # loss, p = self._opt_perm_loss(preds[i], targets[i])
            # for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = self.stack(losses)
        return loss, perms


# def get_si_snr_with_pitwrapper(source, estimate_source):
#     """This function wraps si_snr calculation with the speechbrain pit-wrapper.
#
#     Arguments:
#     ---------
#     source: [B, T, C],
#         Where B is the batch size, T is the length of the sources, C is
#         the number of sources the ordering is made so that this loss is
#         compatible with the class PitWrapper.
#
#     estimate_source: [B, T, C]
#         The estimated source.
#
#     Example:
#     ---------
#     >>> x = torch.arange(600).reshape(3, 100, 2)
#     >>> xhat = x[:, :, (1, 0)]
#     >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
#     >>> print(si_snr)
#     tensor([135.2284, 135.2284, 135.2284])
#     """
#
#     pit_si_snr = PitWrapper(cal_si_snr)
#     loss, perms = pit_si_snr(source, estimate_source)
#
#     return loss

class get_si_snr_with_pitwrapper(nn.Cell):
    """自定义损失函数 get_si_snr_with_pitwrapper"""

    def __init__(self):
        """初始化"""
        super(get_si_snr_with_pitwrapper, self).__init__()
        self.pit_si_snr = PitWrapper()

    def construct(self, source, estimate_source):
        """调用算子"""
        loss, perms = self.pit_si_snr(source, estimate_source)

        return loss


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    keep_reducesum = ops.ReduceSum(keep_dims=True)
    EPS = 1e-8
    # assert source.shape == estimate_source.shape
    # device = estimate_source.device.type

    # source_lengths = torch.tensor(
    #     [estimate_source.shape[0]] * estimate_source.shape[1], device=device
    # )
    source_lengths = Tensor([estimate_source.shape[0]] * estimate_source.shape[1])
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # num_samples = (
    #     source_lengths.contiguous().reshape(1, -1, 1).float()
    # )  # [1, B, 1]
    num_samples = (
        source_lengths.copy().reshape((1, -1, 1)).astype("float32")
    )  # [1, B, 1]
    mean_target = keep_reducesum(source, 0) / num_samples
    mean_estimate = (
            keep_reducesum(estimate_source, 0) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = keep_reducesum(s_estimate * s_target, 0)  # [1, B, C]
    s_target_energy = (keep_reducesum(s_target ** 2, 0) + EPS)  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    reducesum = ops.ReduceSum()
    si_snr_beforelog = reducesum(proj ** 2, 0) / (
            reducesum(e_noise ** 2, 0) + EPS
    )
    si_snr = 10 * np.log10(si_snr_beforelog + EPS)  # [B, C]

    expand_dims = ops.ExpandDims()
    si_snr = expand_dims(si_snr, 0)
    # return -ops.ExpandDims()(si_snr, 0)
    return -si_snr


def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    T, B, _ = source.shape
    # mask = source.new_ones((T, B, 1))
    mask = Tensor(np.ones((T, B, 1)), dtype=source.dtype)
    for i in range(B):
        mask[source_lengths[i]:, i, :] = 0
    return mask

