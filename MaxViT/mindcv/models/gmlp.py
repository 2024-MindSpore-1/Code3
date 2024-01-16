"""
MindSpore implementation of `gMLP`.
Refer to Pay Attention to MLPs.
"""
from functools import partial

import mindspore.common.initializer as init
import numpy as np
from mindspore import dtype as mstype, nn, ops, Parameter, Tensor
from mindspore.common.initializer import Normal

from .helpers import load_pretrained
from .layers import Dropout, DropPath
from .layers.mlp import GatedMlp
from .registry import register_model
from .vit import PatchEmbedding

__all__ = [
    "gMLP",
    "gmlp_s16_224"
]


class SpatialGatingUnit(nn.Cell):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer((gate_dim,))
        # accelerate a lot
        self.proj = nn.Dense(seq_len, seq_len, has_bias=False)
        self.add_bias = Parameter(
            Tensor(np.ones([1, seq_len, 1]), dtype=mstype.float32))
        self.init_weights()

    def init_weights(self):
        self.proj.weight.set_data(init.initializer(Normal(sigma=1e-6), self.proj.weight.shape, self.proj.weight.dtype))
        if self.proj.has_bias:
            self.proj.bias.set_data(init.initializer("ones", self.proj.bias.shape, self.proj.bias.dtype))

    def construct(self, x: Tensor) -> Tensor:
        u, v = x.chunk(2, axis=-1)
        # u, v = ops.split(x, axis=-1, output_num=2)
        v = self.norm(v)
        v = self.proj(v.transpose(0, 2, 1))
        return u * (v.transpose(0, 2, 1) + self.add_bias)


class SpatialGatingBlock(nn.Cell):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(
            self,
            dim,
            seq_len,
            mlp_ratio=4,
            mlp_layer=GatedMlp,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer((dim,))
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class gMLP(nn.Cell):
    r"""gMLP model class, based on
    `"Pay Attention to MLPs" <https://arxiv.org/abs/2105.08050>`_

    Args:
        num_classes (int) : number of classification classes. Default: 1000.
        in_chans (int): number the channels of the input. Default: 3.
        image_size (int): input image size. Default: 224.
        patch_size (int) : size of a single image patch. Default: 16.
        num_blocks (int) : number of SpatialGatingBlocks. Default: 8.
        embed_dim (int) : channels(dimension) of a single embedded patch. Default: 512.
        mlp_ratio (int) : Ratio of mlp hidden dim to embedding dim.
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm.
        act_layer (nn.Cell, optional): Activation layer. Default: nn.GELU.
        drop_rate (float): Pre-classifier dropout rate. Default: 0.0.
        proj_drop_rate (float) - dropout rate after each dense layer. Default: 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.0.
    """

    def __init__(
            self,
            num_classes=1000,
            image_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            input_channels=in_chans,
            embed_dim=embed_dim,
        )
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        # dpr = [x for x in np.linspace(0, drop_path_rate, num_blocks)]  # stochastic depth decay rule
        dpr = [drop_path_rate, ] * num_blocks
        self.blocks = nn.SequentialCell(*[
            SpatialGatingBlock(
                embed_dim,
                self.stem.num_patches,
                mlp_ratio,
                mlp_layer=GatedMlp,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(num_blocks)])
        self.norm = norm_layer((embed_dim,))
        self.head_drop = Dropout(drop_rate)
        self.head = nn.Dense(embed_dim, self.num_classes)
        self._initialize_weights()

    def construct_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def construct_head(self, x, pre_logits: bool = False):
        x = x.mean(axis=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def construct(self, x):
        x = self.construct_features(x)
        x = self.construct_head(x)
        return x

    def _initialize_weights(self):
        self.stem.conv.weight.set_data(
            init.initializer("TruncatedNormal", self.stem.conv.weight.shape, self.stem.conv.weight.dtype))
        self.head.weight.set_data(
            init.initializer("Zero", self.head.weight.shape, self.head.weight.dtype))
        self.head.bias.set_data(
            init.initializer('Zero', self.head.bias.shape, self.head.bias.dtype))


@register_model
def gmlp_s16_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, **kwargs)
    model = gMLP(**model_args)
    if pretrained:
        load_pretrained(model, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def gmlp_t16_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ gMLP-tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, **kwargs)
    model = gMLP(**model_args)
    if pretrained:
        load_pretrained(model, num_classes=num_classes, in_channels=in_channels)
    return model
