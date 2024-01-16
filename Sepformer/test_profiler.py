import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.dataset as ds

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator():
    for i in range(2):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))


def train(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator, ["data", "label"])
    model = ms.Model(net, loss, optimizer)
    model.train(1, data)


if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    ms.set_context(device_id=6)

    # Init Profiler
    # Note that the Profiler should be initialized before model.train
    profiler = ms.Profiler(output_path = './profiler_data')

    # Train Model
    net = Net()
    train(net)

    # Profiler end
    profiler.analyse()
