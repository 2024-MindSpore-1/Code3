from typing import Tuple

from mindspore import Tensor, nn, ops

from ..utils.vd_resnet_cells import ConvNormLayer

__all__ = ['PGHead']


def dummy(*args, **kwargs):
    return


class PGHead(nn.Cell):
    def __init__(self, in_channels: int, num_classes: int = 37, debug: bool = False, **kwargs):  # 36+1
        super().__init__()
        self.tensor_summary = ops.TensorSummary() if debug else dummy
        self.hist_summary = ops.HistogramSummary() if debug else dummy

        # score
        self.conv_f_score1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            act=True)
        self.conv_f_score2 = ConvNormLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            act=True)
        self.conv_f_score3 = ConvNormLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            act=True)

        self.conv1 = nn.Conv2d(
            in_channels=128,
            out_channels=1,
            kernel_size=3)

        # border
        self.conv_f_boder1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            act=True)
        self.conv_f_boder2 = ConvNormLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            act=True)
        self.conv_f_boder3 = ConvNormLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            act=True)

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=4,
            kernel_size=3)

        # character
        self.conv_f_char1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            act=True)
        self.conv_f_char2 = ConvNormLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            act=True)
        self.conv_f_char3 = ConvNormLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            act=True)
        self.conv_f_char4 = ConvNormLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            act=True)
        self.conv_f_char5 = ConvNormLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            act=True)

        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=3)

        # direction
        self.conv_f_direc1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            act=True)
        self.conv_f_direc2 = ConvNormLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            act=True)
        self.conv_f_direc3 = ConvNormLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            act=True)

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=3)

    def construct(self, x: Tensor, **kwargs) -> Tuple[Tensor]:
        f_score = self.conv_f_score1(x)
        f_score = self.conv_f_score2(f_score)
        f_score = self.conv_f_score3(f_score)
        f_score = self.conv1(f_score)
        f_score = ops.sigmoid(f_score)  # [1, 1, 128, 128]

        # border TBO
        f_border = self.conv_f_boder1(x)
        f_border = self.conv_f_boder2(f_border)
        f_border = self.conv_f_boder3(f_border)
        f_border = self.conv2(f_border)  # [1, 4, 128, 128]

        # character TCC
        f_char = self.conv_f_char1(x)
        f_char = self.conv_f_char2(f_char)
        f_char = self.conv_f_char3(f_char)
        f_char = self.conv_f_char4(f_char)
        f_char = self.conv_f_char5(f_char)
        f_char = self.conv3(f_char)  # [1, 37, 128, 128]
        if not self.training:
            f_char = ops.softmax(f_char, 1)

        # direction TDO
        f_direction = self.conv_f_direc1(x)
        f_direction = self.conv_f_direc2(f_direction)
        f_direction = self.conv_f_direc3(f_direction)
        f_direction = self.conv4(f_direction)  # [1, 2, 128, 128]

        self.tensor_summary('f_score', f_score)
        self.tensor_summary('f_border', f_border)
        self.tensor_summary('f_char', f_char)
        self.tensor_summary('f_direction', f_direction)

        self.hist_summary('f_score', f_score)
        self.hist_summary('f_border', f_border)
        self.hist_summary('f_char', f_char)
        self.hist_summary('f_direction', f_direction)

        return f_score, f_border, f_char, f_direction


class PGHead2(nn.Cell):
    def __init__(self, in_channels: int, num_classes: int = 37, **kwargs):  # 36+1
        super().__init__()
        # score
        self.conv_f_score1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            act=True)
        self.conv_f_score2 = ConvNormLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            act=True)
        self.conv_f_score3 = ConvNormLayer(
            in_channels=64,
            out_channels=32,  # 128 -> 32
            kernel_size=1,
            act=True)

        self.conv1 = nn.Conv2d(
            in_channels=32,  # 128 -> 32
            out_channels=1,
            kernel_size=3)

        # border
        self.conv_f_boder1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            act=True)
        self.conv_f_boder2 = ConvNormLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            act=True)
        self.conv_f_boder3 = ConvNormLayer(
            in_channels=64,
            out_channels=32,  # 128 -> 32
            kernel_size=1,
            act=True)

        self.conv2 = nn.Conv2d(
            in_channels=32,  # 128 -> 32
            out_channels=4,
            kernel_size=3)

        # character
        self.conv_f_char1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=1,
            act=True)
        self.conv_f_char2 = ConvNormLayer(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            act=True)
        self.conv_f_char3 = ConvNormLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            act=True)
        self.conv_f_char4 = ConvNormLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            act=True)
        self.conv_f_char5 = ConvNormLayer(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            act=True)

        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=3)

        # direction
        self.conv_f_direc1 = ConvNormLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=1,
            act=True)
        self.conv_f_direc2 = ConvNormLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            act=True)
        self.conv_f_direc3 = ConvNormLayer(
            in_channels=64,
            out_channels=32,  # 128 -> 32
            kernel_size=1,
            act=True)

        self.conv4 = nn.Conv2d(
            in_channels=32,  # 128 -> 32
            out_channels=2,
            kernel_size=3)

    def construct(self, x: Tensor, **kwargs) -> Tuple[Tensor]:
        f_score = self.conv_f_score1(x)
        f_score = self.conv_f_score2(f_score)
        f_score = self.conv_f_score3(f_score)
        f_score = self.conv1(f_score)
        f_score = ops.sigmoid(f_score)  # [1, 1, 128, 128]

        # border TBO
        f_border = self.conv_f_boder1(x)
        f_border = self.conv_f_boder2(f_border)
        f_border = self.conv_f_boder3(f_border)
        f_border = self.conv2(f_border)  # [1, 4, 128, 128]

        # character TCC
        f_char = self.conv_f_char1(x)
        f_char = self.conv_f_char2(f_char)
        f_char = self.conv_f_char3(f_char)
        f_char = self.conv_f_char4(f_char)
        f_char = self.conv_f_char5(f_char)
        f_char = self.conv3(f_char)  # [1, 37, 128, 128]
        if not self.training:
            f_char = ops.softmax(f_char, 1)

        # direction TDO
        f_direction = self.conv_f_direc1(x)
        f_direction = self.conv_f_direc2(f_direction)
        f_direction = self.conv_f_direc3(f_direction)
        f_direction = self.conv4(f_direction)  # [1, 2, 128, 128]

        return f_score, f_border, f_char, f_direction
