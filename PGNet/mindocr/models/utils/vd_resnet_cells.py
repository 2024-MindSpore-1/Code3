from mindspore import Tensor, nn


class ConvNormLayer(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = -1,
                 act: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            group=groups,
            pad_mode='pad' if padding > 0 else 'same',
            padding=padding if padding > 0 else 0)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act_func = nn.ReLU()
        self.act = act

    def construct(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        y = self.norm(y)
        if self.act:
            y = self.act_func(y)
        return y


class DeConvNormLayer(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2,
                 groups: int = 1,
                 act: bool = False):
        super().__init__()
        self.deconv = nn.Conv2dTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            group=groups)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act_func = nn.ReLU()
        self.act = act

    def construct(self, x):
        y = self.deconv(x)
        y = self.norm(y)
        if self.act:
            y = self.act_func(y)
        return y


class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 shortcut: bool = True,
                 pytorch_like: bool = False,
                 **kwargs):
        super().__init__()

        self.conv0 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            act=True)
        self.conv1 = ConvNormLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=stride,
            padding=1 if pytorch_like else -1,
            act=True)
        self.conv2 = ConvNormLayer(
            in_channels=channels,
            out_channels=channels * self.expansion,
            kernel_size=1)

        self.shortcut = shortcut
        if not shortcut:
            self.short = ConvNormLayer(
                in_channels=in_channels,
                out_channels=channels * self.expansion,
                kernel_size=1,
                stride=stride)

        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)

        if self.shortcut:
            identity = x
        else:
            identity = self.short(x)

        out += identity
        out = self.relu(out)
        return out
