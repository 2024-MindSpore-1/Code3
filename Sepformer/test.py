import torch
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
from mindspore import context
from mindspore import dtype as mstype
import mindspore.numpy as mnp

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(device_id=2)

    # print("test x.transpose")
    # x = Tensor(np.ones((1, 2, 3), dtype=np.float32))
    # x = x.transpose([0, 2, 1])
    # print(x.shape)
    # print(torch)
    # x = torch.randn(1, 2, 3)
    # x = x.transpose(2, 1)
    # print(x.size())

    # print("test view")
    # a = Tensor(np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32))
    # output = a.view((-1, 2))
    # print(output.shape)
    # print(output)

    # print("test ops.Zeros")
    # zeros = ops.Zeros()
    # output = zeros((2, 2), mindspore.float32)
    # print(output.shape)
    # print(output.dtype)
    # pad = mindspore.Tensor(np.zeros((1, 2, 3)), dtype=output.dtype)
    # print(pad.dtype)


    # print("Tensor.copy()")
    # a = Tensor(np.ones((3, 3)).astype("float32"))
    # output = a.copy()
    # output[0][0] = 2
    # print(a)
    # print(output)

    # print("test np.tile")
    # pred = Tensor(np.ones((1, 2, 3)).astype("float32"))
    # n_sources = pred.shape[-1]
    # # pred = pred.unsqueeze(-2).repeat(
    # #     *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
    # # )
    # expand_dims = ops.ExpandDims()
    # # pred = expand_dims(pred, -2)
    # # print(pred.shape)
    # # print((*[1 for x in range(len(pred.shape) - 1)], n_sources, 1))
    # pred = mnp.tile(expand_dims(pred, -2), (*[1 for x in range(len(pred.shape) - 1)], n_sources, 1))
    # print(pred.shape)

    # print("test get_lr()")
    # from mindspore import nn
    #
    # class Net(nn.Cell):
    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.conv = nn.Conv2d(3, 64, 3)
    #         self.bn = nn.BatchNorm2d(64)
    #         self.relu = nn.ReLU()
    #
    #     def construct(self, x):
    #         x = self.conv(x)
    #         x = self.bn(x)
    #         x = self.relu(x)
    #         return x
    #
    # net = Net()
    # conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    # no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    # group_params = [{'params': conv_params, 'lr': 0.05},
    #                 {'params': no_conv_params, 'lr': 0.01}]
    # optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    # lr = optim.get_lr()
    # conv_lr = optim.get_lr_parameter(conv_params)
    # for i in range(len(lr)):
    #     print(lr[i].asnumpy())
    # print(conv_lr[0].asnumpy())

    # print("uniform_int = ops.UniformInt()")
    # uniform_int = ops.UniformInt()
    # samp_index = uniform_int((0,), Tensor(1, mstype.int32), Tensor(3, mstype.int32))
    # print(type(samp_index.item(0)))
    # print(samp_index)
    # print(samp_index.shape)
    #
    # print("test uniform_real")
    # # shape = (2,)
    # # samp_index = Tensor(2, mstype.int32)
    # print(type(samp_index))
    # a = samp_index.asnumpy()
    # print("type(a)=", type(a))
    # shape = (samp_index.asnumpy(),)
    # print("shape = ", shape)
    # uniform_real = ops.UniformReal()
    # output = uniform_real(shape)
    # result = output.shape
    # print(result)


    # print("mindspore.dtype_to_pytype(type_)")
    # func = mindspore.dtype_to_pytype(mstype.int32)
    # shape = Tensor([2], mstype.int32)
    # print("shape", shape.shape)
    # shape = func(shape)
    # print(type(shape))
    # print(shape)

    # print("test transpose")
    # print("torch")
    # x = torch.randn(1, 2, 3)
    # x = x.transpose(1, 2)
    # print(len(x.shape))
    # print(x.shape)
    # print("mindspore")
    # x = Tensor(np.ones((1, 2, 3), dtype=np.float32))
    # x = x.transpose(0, 2, 1)
    # print(x.shape)

    # print("test shape")
    # waveforms = Tensor(np.ones((1, 2, 3), dtype=np.float32))
    # print(waveforms.shape[1])
    # batch_size, num_channels, wave_len = waveforms.shape
    # print(batch_size)
    # print(num_channels)
    # print(wave_len)

    # print("test ops.Pad")
    # input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    # pad_op = ops.Pad(((1, 2), (2, 1)))
    # output = pad_op(input_x)
    # print(output.shape)
    # print(output)

    # print("test ops.Less()")
    # x = Tensor(np.array([1, 2, 3]), mindspore.int32)
    # # y = Tensor(np.array([1, 1, 4]), mindspore.int32)
    # y = Tensor(np.array([2]), mindspore.int32)
    # less = ops.Less()
    # output = less(x, y)
    # print(output)

    # print("test .dtype()")
    # x = Tensor(1, mindspore.int32)
    # print(x.dtype)
    # y = x.astype("int64", copy=False)
    # # y[0, 0] = Tensor(2, mindspore.int64)
    # print(x.dtype)
    # print(x)
    # print(y.dtype)
    # print(y)
    # z = torch.ones(1, 2)
    # zz = z.long()
    # zz[0][0] = 2
    #
    # print(z)
    # print(type(z))
    # print(z)
    # print(type(zz))
    # print(zz)

    # print("test ops.Ones()")
    # ones = ops.Ones()
    # output = ones(1, mindspore.float32)
    # print(output)

    print("test np.tile")
    import mindspore.numpy as np
    from mindspore import Tensor
    a = np.array([[0, 2, 1], [3, 4, 5]])
    b = np.tile(a, 2)
    print(b.shape)
    source_lengths = Tensor([b.shape[0]] * b.shape[1])
    print(type(source_lengths))
    print(source_lengths)












