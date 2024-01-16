import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class ChannelAtt(nn.Cell):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.fc = nn.SequentialCell([
                nn.Dense(channel, channel//reduction),
                nn.ReLU(),
                nn.Dense(channel//reduction, channel),
                nn.Sigmoid()])

    def construct(self, x):
        b, c, height, width = x.shape
        y = nn.AvgPool2d(kernel_size=(height, width), stride=2)(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CountingDecoder(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.trans_layer = nn.SequentialCell([
            nn.Conv2d(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size//2, pad_mode='pad'),
            nn.BatchNorm2d(512, momentum=0.1)])
        self.channel_att = ChannelAtt(512, 16)
        self.pred_layer = nn.SequentialCell([
            nn.Conv2d(512, self.out_channel, kernel_size=1),
            nn.Sigmoid()])

    def construct(self, x, mask):
        b, c, h, w = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)
        if mask is not None:
            x = x * mask
        x = x.view(b, self.out_channel, -1)
        x1 = x.sum(-1)
        return x1, x.view(b, self.out_channel, h, w)
