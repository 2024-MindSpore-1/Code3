from mindspore import ops
import mindspore.common.dtype as mstype
import mindspore
from mindspore import Tensor

if __name__ == '__main__':
    from mindspore import context
    target = "Ascend"
    device_id = 5

    # init context
    # context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    context.set_context(device_id=device_id)


    # # test src.augment.TimeDomainSpecAugment
    # >>> inputs = torch.randn([10, 16000])
    # >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    # >>> feats = feature_maker(inputs, torch.ones(10))
    # >>> feats.shape
    # torch.Size([10, 12800])
    from mindspore import ops
    from src.augment import TimeDomainSpecAugment
    import mindspore


    feature_maker = TimeDomainSpecAugment(speeds=[80])
    shape = (10, 16000)
    stdnormal = ops.StandardNormal(seed=2)
    inputs = stdnormal(shape)
    ones = ops.Ones()
    targets = ones((10,), mindspore.float32)
    feats = feature_maker(inputs, targets)
    print(feats.shape)
    # # done
    # # PYNATIVE_MODE OK (10, 12800)

    # # # test src.speech_augmentation
    # from dataprocess.dataio import read_audio
    # from src.speech_augmentation import SpeedPerturb, Resample
    # #
    # # signal = read_audio('/old/gxl/sepformer/example1.wav')
    # shape = (52173,)
    # stdnormal = ops.StandardNormal(seed=2)
    # signal = stdnormal(shape)
    # print("signal = ", signal.shape)
    # # print(signal)
    # signal = Tensor(signal, mindspore.float32)
    # # signal = signal.unsqueeze(0) # [batch, time, channels]
    # expand_dims = ops.ExpandDims()
    # signal = expand_dims(signal, 0)
    # #
    # # # test Resample
    # resampler = Resample(orig_freq=16000, new_freq=8000)
    # resampled = resampler(signal)
    # print("signal.shape = ", signal.shape)
    # # torch.Size([1, 52173])
    # print("resampled.shape = ", resampled.shape)
    # # torch.Size([1, 26087])
    # # # done PYNATIVE_MODE OK
    # # # signal.shape = (1, 52173)
    # # # resampled.shape = (1, 26087)

    # # test SpeedPerturb
    # perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    # perturbed = perturbator(signal)
    # print("perturbed.shape = ", perturbed.shape)
    # # torch.Size([1, 46956])
    # # done perturbed.shape = (1, 46956) PYNATIVE_MODE OK

    # # test src.dual_path.Decoder
    # from src.dual_path import Decoder
    # """A decoder layer that consists of ConvTranspose1d.
    # >>> x = torch.randn(2, 100, 1000)
    # >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    # >>> h = decoder(x)
    # >>> h.shape
    # torch.Size([2, 1003])
    # """
    # shape = (2, 100, 1000)
    # stdnormal = ops.StandardNormal(seed=2)
    # x = stdnormal(shape)
    # decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    # h = decoder(x)
    # print(h.shape)
    # # (2, 1000) 与 torch.Size([2, 1003]) 有差别

    # """
    # >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    # >>> targets = torch.rand((2, 32, 4))
    # >>> p = (3, 0, 2, 1)
    # >>> predictions = targets[..., p]
    # >>> loss, opt_p = pit_mse(predictions, targets)
    # >>> loss
    # tensor([0., 0.])
    # """

    # from itertools import permutations
    # # loss_mat = torch.randn(4, 4)
    # shape = (4, 4)
    # stdnormal = ops.StandardNormal(seed=2)
    # loss_mat = stdnormal(shape)
    # print(loss_mat)
    # # 对tuple进行permutations排列组合
    # shape = tuple(range(loss_mat.shape[0]))
    # print(shape)
    # for p in permutations(range(loss_mat.shape[0])):
    #     c_loss = loss_mat[shape, p].mean()
    #     print(c_loss)


    # # #  test src.losses.PitWrapper
    # # from src.losses import PitWrapper
    # # pit_mse = PitWrapper()
    # # shape = (2, 32, 4)
    # # stdnormal = ops.StandardNormal(seed=2)
    # # targets = stdnormal(shape)
    # # p = (3, 0, 2, 1)
    # # predictions = targets[..., p]
    # # loss, opt_p = pit_mse(predictions, targets)
    # # print(loss)
    # # # [-95.32259  -94.726585]
    #
    # '''
    # >>> x = torch.arange(600).reshape(3, 100, 2)
    # >>> xhat = x[:, :, (1, 0)]
    # >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    # >>> print(si_snr)
    # tensor([135.2284, 135.2284, 135.2284])
    # '''
    # # test src.losses.get_si_snr_with_pitwrapper
    # from src.testlosses import get_si_snr_with_pitwrapper
    # import mindspore.numpy as np
    #
    # x = np.arange(600)
    # x = ops.Cast()(x, mindspore.float32)
    # x = ops.reshape(x, (3, 100, 2))
    # xhat = x[:, :, (1, 0)]
    # print("x = ", x.shape)
    # print(type(x))
    # print("xhat = ", xhat.shape)
    # print(type(xhat))
    # get_si_snr_with_pitwrapper = get_si_snr_with_pitwrapper()
    # si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    # print(si_snr)

