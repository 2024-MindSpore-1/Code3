"""
MindSpore implementation of `ResNet`.
Refer to Deep Residual Learning for Image Recognition.
"""
from typing import List, Optional, Type, Union

import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common import initializer as init

from mindcv.models.helpers import build_model_with_cfg
from mindcv.models.layers.drop_path import DropPath
from mindcv.models.layers.helpers import to_2tuple
from mindcv.models.layers.pooling import GlobalAvgPooling
from mindcv.models.registry import register_model
from mindcv.models.resnet import Bottleneck

BatchNorm2d = nn.BatchNorm2d

__all__ = [
    "BoTNet",
    "botnet50",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "botnet50": _cfg(),
}


def expand_dim(t, axis, repeats):
    """
    Expand dims for t at dim to k
    """
    t = t.expand_dims(axis=axis)
    t = t.repeat(repeats, axis)
    return t


def rel_to_abs(x):
    """
    x: [B, Nh * H, L, 2L - 1]
    Convert relative position between the key and query to their absolute position respectively.
    Tensowflow source code in the appendix of: https://arxiv.org/pdf/1904.09925.pdf
    """
    B, Nh, L, _ = x.shape
    # pad to shift from relative to absolute indexing
    col_pad = ops.zeros((B, Nh, L, 1), x.dtype)
    x = ops.concat((x, col_pad), axis=3)
    flat_x = ops.reshape(x, (B, Nh, L * 2 * L))
    flat_pad = ops.zeros((B, Nh, L - 1), x.dtype)
    flat_x = ops.concat((flat_x, flat_pad), axis=2)
    # Reshape and slice out the padded elements
    final_x = ops.reshape(flat_x, (B, Nh, L + 1, 2 * L - 1))
    # return ops.slice(final_x, (0, 0, 0, L - 1), (-1, -1, L, -1))
    return final_x[:, :, :L, L - 1:]


def relative_logits_1d(q, rel_k):
    """
    q: [B, Nh, H, W, d]
    rel_k: [2W - 1, d]
    Computes relative logits along one dimension.
    The details of relative position is explained in: https://arxiv.org/pdf/1803.02155.pdf
    """
    B, Nh, H, W, D = q.shape

    rel_k = ops.cast(rel_k, q.dtype)
    rel_logits = ops.MatMul(transpose_b=True)(q.reshape(-1, D), rel_k)
    # Collapse height and heads
    rel_logits = ops.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = ops.reshape(rel_logits, (-1, Nh, H, W, W))
    tile = ops.zeros((1, 1, 1, H, 1, 1), rel_logits.dtype)
    rel_logits = rel_logits.expand_dims(axis=3) + tile
    # rel_logits = expand_dim(rel_logits, axis=3, repeats=H)
    return rel_logits


class RelativePostionEmbedding2D(nn.Cell):
    def __init__(self, height, width, num_heads):
        super().__init__()
        scale = num_heads ** -0.5
        self.height = height
        self.width = width
        self.rel_height_bias = Parameter(Tensor(np.random.randn(height * 2 - 1, num_heads) * scale, mstype.float32))
        self.rel_width_bias = Parameter(Tensor(np.random.randn(width * 2 - 1, num_heads) * scale, mstype.float32))

    def construct(self, q):
        height = self.height
        width = self.width
        B, N, _, D = q.shape
        q = q.reshape(B, N, height, width, D)
        rel_logits_w = relative_logits_1d(q, self.rel_width_bias)
        rel_logits_w = rel_logits_w.transpose(0, 1, 2, 4, 3, 5)
        shape = rel_logits_w.shape
        rel_logits_w = rel_logits_w.reshape(shape[0], shape[1], shape[2] * shape[3], -1)

        q = q.transpose(0, 1, 3, 2, 4)
        rel_logits_h = relative_logits_1d(q, self.rel_height_bias)
        rel_logits_h = rel_logits_h.transpose(0, 1, 4, 2, 5, 3)
        shape = rel_logits_h.shape
        rel_logits_h = rel_logits_h.reshape(shape[0], shape[1], shape[2] * shape[3], -1)
        return rel_logits_w + rel_logits_h


class BoTBlock(nn.Cell):
    expansion: int = 4

    def __init__(
            self,
            dim,
            fmap_size,
            dim_out,
            stride=1,
            num_heads=4,
            proj_factor=4,
            dim_qk=128,
            dim_v=128,
            activation=nn.ReLU,
            drop_path_rate=0.,
            gamma_init='ones',
    ):
        """
        dim: channels in feature map
        dim_out: output channels for feature map
        """
        super().__init__()
        if dim != dim_out or stride != 1:
            self.down_sample = nn.SequentialCell(
                nn.Conv2d(dim, dim_out, kernel_size=1, stride=stride, has_bias=False),
                BatchNorm2d(dim_out),
                activation(),
            )
        else:
            self.down_sample = nn.Identity()

        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        bottleneck_dimension = dim_out // proj_factor  # from 2048 to 512
        attn_dim_out = num_heads * dim_v

        self.net = nn.SequentialCell(
            nn.Conv2d(dim, bottleneck_dimension, kernel_size=1, stride=1, has_bias=False),
            BatchNorm2d(bottleneck_dimension),
            activation(),
            MultiHeadAttention(
                dim=bottleneck_dimension,
                feature_size=fmap_size,
                num_heads=num_heads,
                dim_qk=dim_qk,
                dim_v=dim_v,
            ),
            nn.AvgPool2d((2, 2), pad_mode='same', stride=2) if stride == 2 else nn.Identity(),  # same padding
            BatchNorm2d(attn_dim_out),
            activation(),
            nn.Conv2d(attn_dim_out, dim_out, kernel_size=1, stride=1, has_bias=False),
            BatchNorm2d(dim_out, gamma_init=gamma_init),
        )
        self.activation = activation()

    def construct(self, x):
        identity = self.down_sample(x)
        x = self.net(x)
        x = x + self.drop_path(identity)
        x = self.activation(x)
        return x


class MultiHeadAttention(nn.Cell):
    def __init__(self, dim, feature_size, num_heads=4, dim_qk=128, dim_v=128):
        """
        dim: number of channels of feature map
        feature_size: [H, W]
        dim_qk: vector dimension for q, k
        dim_v: vector dimension for v (not necessarily the same with q, k)
        """
        super().__init__()
        self.scaling = dim_qk ** -0.5
        self.num_heads = num_heads
        self.out_channels_qk = num_heads * dim_qk
        self.out_channels_v = num_heads * dim_v
        # 1*1 conv to compute q, k
        self.q = nn.Dense(dim, self.out_channels_qk, has_bias=False)
        self.k = nn.Dense(dim, self.out_channels_qk, has_bias=False)
        # 1*1 conv to compute v
        self.v = nn.Dense(dim, self.out_channels_v, has_bias=False)
        self.softmax = nn.Softmax(axis=-1)

        height, width = feature_size
        self.pos_emb = RelativePostionEmbedding2D(height, width, dim_qk)

    def construct(self, x):
        """
        x: [B, d_in, H, W]
        Output: [B, H, W, head * d_v]
        """
        num_heads = self.num_heads
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(0, 2, 1)

        q, k, v = self.q(x), self.k(x), self.v(x)
        # [B, heads, H x W, dim_q | dim_k | dim_v]
        q = q.reshape(B, -1, num_heads, self.out_channels_qk // num_heads).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, num_heads, self.out_channels_qk // num_heads).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, num_heads, self.out_channels_v // num_heads).transpose(0, 2, 1, 3)

        # [B, heads, H x W, H x W]
        attn = ops.BatchMatMul(transpose_b=True)(q, k) * self.scaling
        attn += self.pos_emb(q)
        attn = self.softmax(attn)
        # [B, head, H x W, dim_v]
        attn_out = ops.transpose(ops.BatchMatMul()(attn, v), (0, 2, 1, 3))
        attn_out = attn_out.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        return attn_out


class BoTStack(nn.Cell):
    def __init__(
            self,
            dim,
            fmap_size,
            drop_path_rate: List[float],
            dim_out=2048,
            num_heads=4,
            proj_factor=4,
            num_layers=3,
            stride=2,
            dim_qk=128,
            dim_v=128,
            activation=nn.ReLU,
    ):
        """
        dim: channels in feature map
        fmap_size: [H, W]
        """
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out

            fmap_divisor = 2 if stride == 2 and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                BoTBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    stride=stride if is_first else 1,
                    num_heads=num_heads,
                    proj_factor=proj_factor,
                    dim_qk=dim_qk,
                    dim_v=dim_v,
                    activation=activation,
                    drop_path_rate=drop_path_rate.pop(0),
                    gamma_init='ones' if i == 0 else 'zeros'
                )
            )

        self.net = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.net(x)


class BoTNet(nn.Cell):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        block: block of botnet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
    """

    def __init__(
            self,
            layers: List[int],
            num_classes: int = 1000,
            in_channels: int = 3,
            groups: int = 1,
            base_width: int = 64,
            norm: Optional[nn.Cell] = None,
            drop_path_rate: float = 0.
    ) -> None:
        super().__init__()
        if norm is None:
            norm = BatchNorm2d

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.drop_path_rates = list(np.linspace(0, drop_path_rate, sum(layers)))

        self.conv1 = nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode="pad", padding=3)
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.feature_info = [dict(chs=self.input_channels, reduction=2, name="relu")]
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.feature_info.append(dict(chs=Bottleneck.expansion * 64, reduction=4, name="layer1"))
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.feature_info.append(dict(chs=Bottleneck.expansion * 128, reduction=8, name="layer2"))
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.feature_info.append(dict(chs=Bottleneck.expansion * 256, reduction=16, name="layer3"))
        # self.layer4 = self._make_layer(BoTBlock, 512, layers[3], stride=2)
        assert len(self.drop_path_rates) == layers[3]
        self.layer4 = BoTStack(fmap_size=to_2tuple(224 // 16), dim=Bottleneck.expansion * 256,
                               dim_out=BoTBlock.expansion * 512, num_layers=layers[3], stride=2,
                               drop_path_rate=self.drop_path_rates)
        self.feature_info.append(dict(chs=BoTBlock.expansion * 512, reduction=32, name="layer4"))

        self.pool = GlobalAvgPooling()
        self.num_features = 512 * Bottleneck.expansion
        self.classifier = nn.Dense(self.num_features, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer('zeros', cell.bias.shape, cell.bias.dtype))
            # elif isinstance(cell, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            #     cell.gamma.set_data(init.initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            #     cell.beta.set_data(init.initializer('zeros', cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(mode='fan_in', nonlinearity='sigmoid'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def _make_layer(
            self,
            block: Type[Union[Bottleneck]],
            channels: int,
            block_nums: int,
            stride: int = 1,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                nn.Conv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
                self.norm(channels * block.expansion)
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
                drop_path_rate=self.drop_path_rates.pop(0)
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm,
                    drop_path_rate=self.drop_path_rates.pop(0)
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_botnet(pretrained=False, **kwargs):
    return build_model_with_cfg(BoTNet, pretrained, **kwargs)


@register_model
def botnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers BoTNet model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["botnet50"]
    model_args = dict(layers=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)
    return _create_botnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


if __name__ == "__main__":
    model = botnet50()
    data = Tensor(np.random.randn(4, 3, 224, 224), mstype.float32)
    print(model(data).shape)
    print(sum([p.size for p in model.get_parameters()]))
