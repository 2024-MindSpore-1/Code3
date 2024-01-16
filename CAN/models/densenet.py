import math

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

# DenseNet-B
class Bottleneck(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels,momentum=0.1)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(growthRate,momentum=0.1)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, pad_mode = 'pad')
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(keep_prob=0.8)

    def construct(self, x):
        relu = nn.ReLU()
        out = self.conv1(x)
        out = relu(self.bn1(out))
        if self.use_dropout:
            out = self.dropout(out)
        out = relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), 1)
        return out


# single layer
class SingleLayer(nn.Cell):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels,momentum=0.1)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(keep_prob=0.8)

    def forward(self, x):
        relu = nn.ReLU()
        out = self.conv1(relu(x))
        if self.use_dropout:
            out = self.dropout(out)
        out = ops.concat((x, out), 1)
        return out


# transition layer
class Transition(nn.Cell):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels,momentum=0.1)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(keep_prob=0.8)

    def construct(self, x):
        relu = nn.ReLU()
        out = relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        avgpool = ops.AvgPool(kernel_size=2, strides=2,pad_mode='same')
        out = avgpool(out)
        return out


class DenseNet(nn.Cell):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate']
        reduction = params['densenet']['reduction']
        bottleneck = params['densenet']['bottleneck']
        use_dropout = params['densenet']['use_dropout']

        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, pad_mode='pad',stride=2)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.SequentialCell(layers)

    def construct(self, x):
        relu = nn.ReLU()
        out = self.conv1(x)
        out = relu(out)
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        out = maxpool(out)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out
