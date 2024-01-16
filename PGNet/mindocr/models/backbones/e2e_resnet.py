from typing import List, Type

from mindspore import Tensor, __version__, nn, ops

from mindocr.models.backbones.mindcv_models.resnet import ResNet

from ..utils.vd_resnet_cells import Bottleneck, ConvNormLayer
from ._registry import register_backbone, register_backbone_class

# from mindocr.models.backbones.mindcv_models.utils import load_pretrained


__all__ = ['E2EResNet', 'e2e_resnet50']


@register_backbone_class
class E2EResNet(ResNet):
    def __init__(self,
                 block: Type[Bottleneck],
                 in_channels: int = 3,
                 layers: int = 50,
                 pytorch_like: bool = False,
                 **kwargs):
        super(ResNet, self).__init__()

        self.pytorch_like = pytorch_like

        self.layers = layers
        supported_layers = {50}
        assert layers in supported_layers, \
            f"supported layers are {supported_layers} but input layer is {layers}"

        self.depth = (3, 4, 6, 3, 3)
        self.num_channels = (64, 256, 512, 1024, 2048)
        self.num_filters = (64, 128, 256, 512, 512)

        # 当 pad_mode == same
        if pytorch_like:  # 优先填充左上
            self.conv1_1 = ConvNormLayer(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, act=True)
            self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:  # mindspore 和 tensorflow 优先填充右下
            self.conv1_1 = ConvNormLayer(in_channels, out_channels=64, kernel_size=7, stride=2, act=True)
            self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.out_channels = [3, 64]

        self.layer1 = self._make_layer(block, block_id=0)
        self.layer2 = self._make_layer(block, block_id=1)
        self.layer3 = self._make_layer(block, block_id=2)
        self.layer4 = self._make_layer(block, block_id=3)
        self.layer5 = self._make_layer(block, block_id=4)

        self._initialize_weights()

    def _make_layer(self,
                    block: Type[Bottleneck],
                    block_id: int) -> nn.SequentialCell:
        """build model depending on cfgs"""
        input_channels = self.num_channels[block_id]
        output_channels = self.num_filters[block_id]
        layers = [
            block(
                input_channels,
                output_channels,
                stride=2 if block_id != 0 else 1,
                shortcut=False,
                pytorch_like=self.pytorch_like)]

        input_channels = output_channels * block.expansion if block == Bottleneck else 1
        for _ in range(1, self.depth[block_id]):
            layers.append(
                block(
                    input_channels,
                    output_channels,
                    stride=1,
                    shortcut=True,
                    pytorch_like=self.pytorch_like))

        self.out_channels.append(input_channels)

        return nn.SequentialCell(layers)

    def construct(self, inputs: Tensor) -> List[Tensor]:
        x0 = self.conv1_1(inputs)
        if self.pytorch_like:
            if __version__ < "2.0":
                tmp = ops.pad(x0, ((0, 0), (0, 0), (1, 1), (1, 1)))  # ms-1.10
            else:
                tmp = ops.pad(x0, (1, 1, 1, 1))  # ms-2.0
            x1 = self.pool2d_max(tmp)
        else:
            x1 = self.pool2d_max(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return [inputs, x0, x1, x2, x3, x4, x5]


@register_backbone
def e2e_resnet50(pretrained: bool = True, **kwargs) -> E2EResNet:
    model = E2EResNet(Bottleneck, in_channels=3, layers=50, **kwargs)

    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `e2e_resnet50` backbone does not exist.")

    return model
