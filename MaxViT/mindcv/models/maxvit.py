""" MaxVit and CoAtNet Vision Transformer - CNN Hybrids in PyTorch

This is a from-scratch implementation of both CoAtNet and MaxVit in PyTorch.

99% of the implementation was done from papers, however last minute some adjustments were made
based on the (as yet unfinished?) public code release https://github.com/google-research/maxvit

There are multiple sets of models defined for both architectures. Typically, names with a
 `_rw` suffix are my own original configs prior to referencing https://github.com/google-research/maxvit.
These configs work well and appear to be a bit faster / lower resource than the paper.

The models without extra prefix / suffix' (coatnet_0_224, maxvit_tiny_224, etc), are intended to
match paper, BUT, without any official pretrained weights it's difficult to confirm a 100% match.

Papers:

MaxViT: Multi-Axis Vision Transformer - https://arxiv.org/abs/2204.01697
@article{tu2022maxvit,
  title={MaxViT: Multi-Axis Vision Transformer},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  journal={ECCV},
  year={2022},
}

CoAtNet: Marrying Convolution and Attention for All Data Sizes - https://arxiv.org/abs/2106.04803
@article{DBLP:journals/corr/abs-2106-04803,
  author    = {Zihang Dai and Hanxiao Liu and Quoc V. Le and Mingxing Tan},
  title     = {CoAtNet: Marrying Convolution and Attention for All Data Sizes},
  journal   = {CoRR},
  volume    = {abs/2106.04803},
  year      = {2021}
}

Hacked together by / Copyright 2022, Ross Wightman
"""
import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import mindspore as ms
import numpy as np
from mindspore import dtype as mstype, nn, ops, Parameter, Tensor

from mindcv.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindcv.models.convnext import ConvNextLayerNorm
from mindcv.models.helpers import load_pretrained
from mindcv.models.layers import Dropout, DropPath, GlobalAvgPooling, make_divisible
from mindcv.models.layers.helpers import to_2tuple
from mindcv.models.layers.mlp import Mlp
from mindcv.models.layers.squeeze_excite import SqueezeExcite
from mindcv.models.layers.weight_init import normal_, trunc_normal_tf_, xavier_normal_, xavier_uniform_, zeros_
from mindcv.models.poolformer import ConvMlp
from mindcv.models.registry import register_model

__all__ = ['MaxxVitCfg', 'MaxxVitConvCfg', 'MaxxVitTransformerCfg', 'MaxxVit',
           'maxvit_tiny_tf_224', 'maxvit_tiny_tf_384', 'maxvit_tiny_tf_512',
           'maxvit_small_tf_224', 'maxvit_small_tf_384', 'maxvit_small_tf_512',
           'maxvit_base_tf_224', 'maxvit_base_tf_384', 'maxvit_base_tf_512',
           'maxvit_large_tf_224', 'maxvit_large_tf_384', 'maxvit_large_tf_512']

_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    celu=nn.CELU,
    gelu=nn.GELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    gelu_tanh=nn.GELU,
)

_NORM_MAP = dict(
    batchnorm=nn.BatchNorm1d,
    batchnorm2d=nn.BatchNorm2d,
    batchnorm1d=nn.BatchNorm1d,
    layernorm=nn.LayerNorm,
    layernorm2d=ConvNextLayerNorm,
)


@dataclass
class MaxxVitTransformerCfg:
    dim_head: int = 32
    head_first: bool = True  # head ordering in qkv channel dim
    expand_ratio: float = 4.0
    expand_first: bool = True
    shortcut_bias: bool = True
    attn_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pool_type: str = 'avg2'
    rel_pos_type: str = 'bias'
    rel_pos_dim: int = 512  # for relative position types w/ MLP
    partition_ratio: int = 32
    window_size: Optional[Tuple[int, int]] = None
    grid_size: Optional[Tuple[int, int]] = None
    no_block_attn: bool = False  # disable window block attention for maxvit (ie only grid)
    use_nchw_attn: bool = False  # for MaxViT variants (not used for CoAt), keep tensors in NCHW order
    init_values: Optional[float] = None
    act_layer: str = 'gelu'
    norm_layer: str = 'layernorm2d'
    norm_layer_cl: str = 'layernorm'
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.grid_size is not None:
            self.grid_size = to_2tuple(self.grid_size)
        if self.window_size is not None:
            self.window_size = to_2tuple(self.window_size)
            if self.grid_size is None:
                self.grid_size = self.window_size


@dataclass
class MaxxVitConvCfg:
    block_type: str = 'mbconv'
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    output_bias: bool = True  # bias for shortcut + final 1x1 projection conv
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    padding: str = ''
    attn_early: bool = False  # apply attn between conv2 and norm2, instead of after norm2
    attn_layer: str = 'se'
    attn_act_layer: str = 'silu'
    attn_ratio: float = 0.25
    init_values: Optional[float] = 1e-6  # for ConvNeXt block, ignored by MBConv
    act_layer: str = 'gelu'
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None

    def __post_init__(self):
        # mbconv blocks have different defaults, set in post_init to avoid explicit config args
        assert self.block_type in ('mbconv',)
        use_mbconv = self.block_type == 'mbconv'
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type


@dataclass
class MaxxVitCfg:
    embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    depths: Tuple[int, ...] = (2, 3, 5, 2)
    block_type: Tuple[Union[str, Tuple[str, ...]], ...] = ('C', 'C', 'T', 'T')
    stem_width: Union[int, Tuple[int, int]] = 64
    stem_bias: bool = False
    conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg()
    transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg()
    head_hidden_size: int = None
    weight_init: str = 'vit_eff'


def generate_lookup_tensor(
        length: int,
        max_relative_position: Optional[int] = None,
):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
            Relative position embeddings for distances above this threshold
            are zeroed out.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
            ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    if max_relative_position is None:
        max_relative_position = length - 1
    # Return the cached lookup tensor, otherwise compute it and cache it.
    vocab_size = 2 * max_relative_position + 1
    ret = np.zeros((length, length, vocab_size), np.float32)
    for i in range(length):
        for x in range(length):
            v = x - i + max_relative_position
            if abs(x - i) > max_relative_position:
                continue
            ret[i, x, v] = 1
    return ret


def reindex_2d_einsum_lookup(
        relative_position_tensor,
        height: int,
        width: int,
        height_lookup: Tensor,
        width_lookup: Tensor,
) -> Tensor:
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py

    Args:
        relative_position_tensor: tensor of shape
            [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        height_lookup: one-hot height lookup
        width_lookup: one-hot width lookup
    Returns:
        reindexed_tensor: a Tensor of shape
            [..., height * width, height * width, ...]
    """
    B = relative_position_tensor.shape[0]
    # n h w -> n 1 h 1 w -> n i x w
    # i x h -> 1 i h x 1 -> n i x w

    relative_position_tensor = relative_position_tensor.expand_dims(1).expand_dims(3)
    height_lookup = height_lookup.transpose(0, 2, 1).expand_dims(0).expand_dims(-1)
    relative_position_tensor = ops.cast(relative_position_tensor, mstype.float16)
    height_lookup = ops.cast(height_lookup, mstype.float16)
    reindexed_tensor = ops.mul(x=relative_position_tensor, y=height_lookup).sum(2)

    # n i x w -> n i 1 x 1 w
    # j y w ->   1 1 j 1 y w
    reindexed_tensor = reindexed_tensor.expand_dims(2).expand_dims(4)
    width_lookup = width_lookup.expand_dims(0).expand_dims(1).expand_dims(3)
    reindexed_tensor = ops.cast(reindexed_tensor, mstype.float16)
    width_lookup = ops.cast(width_lookup, mstype.float16)
    reindexed_tensor = ops.mul(x=reindexed_tensor, y=width_lookup).sum(-1)
    area = height * width

    # reindexed_tensor = torch.einsum('nhw,ixh->nixw', relative_position_tensor, height_lookup)
    # reindexed_tensor = torch.einsum('nixw,jyw->nijxy', reindexed_tensor, width_lookup)
    # nhw -> n 1 h w
    # ixh -> 1 i x h
    # n i w x -> n i x w
    # relative_position_tensor = relative_position_tensor.expand_dims(axis=1)
    # height_lookup = height_lookup.expand_dims(axis=0)
    # relative_position_tensor = ops.cast(relative_position_tensor, mstype.float16)
    # height_lookup = ops.cast(height_lookup, mstype.float16)
    # reindexed_tensor = ops.BatchMatMul(True, True)(relative_position_tensor, height_lookup).transpose(0, 1, 3, 2)

    # nixw -> n i 1 x w
    # jyw  -> 1 1 j y w
    # n i j x y
    # reindexed_tensor = reindexed_tensor.expand_dims(axis=2)
    # width_lookup = width_lookup.expand_dims(axis=0).expand_dims(axis=1)
    # reindexed_tensor = ops.cast(reindexed_tensor, mstype.float16)
    # width_lookup = ops.cast(width_lookup, mstype.float16)
    # reindexed_tensor = ops.BatchMatMul(transpose_b=True)(reindexed_tensor, width_lookup)
    # area = height * width
    return reindexed_tensor.reshape(B, area, area)


def reindex_2d_einsum_lookup2(
        relative_position_tensor,
        height: int,
        width: int,
        height_lookup: Tensor,
        width_lookup: Tensor,
) -> Tensor:
    B = relative_position_tensor.shape[0]
    n, h, w = relative_position_tensor.shape
    i, x, h = height, height, height * 2 - 1
    reindexed_tensor = ops.gather(relative_position_tensor, height_lookup, axis=1).reshape(n, i, x, w)
    # reindexed_tensor = relative_position_tensor[:, height_lookup].reshape(n, i, x, w)

    n, i, x, w = reindexed_tensor.shape
    j, y, w = width, width, width * 2 - 1
    reindexed_tensor = ops.gather(reindexed_tensor, width_lookup, axis=-1).reshape(n, i, x, j, y)
    # reindexed_tensor = reindexed_tensor[:, :, :, width_lookup].reshape(n, i, x, j, y)
    reindexed_tensor = reindexed_tensor.transpose(0, 1, 3, 2, 4)
    area = height * width
    return reindexed_tensor.reshape(B, area, area)


class RelPosBiasTf(nn.Cell):
    """ Relative Position Bias Impl (Compatible with Tensorflow MaxViT models)
    Adapted from:
     https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/attention_utils.py
    """

    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        assert prefix_tokens <= 1
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = num_heads

        self.vocab_height = 2 * window_size[0] - 1
        self.vocab_width = 2 * window_size[1] - 1
        self.bias_shape = (self.num_heads, self.vocab_height, self.vocab_width)
        self.relative_position_bias_table = Parameter(Tensor(ops.zeros(self.bias_shape, mstype.float32)))
        self.height_lookup = ops.Argmax(axis=-1)(Tensor(generate_lookup_tensor(window_size[0]), mstype.float32))
        self.width_lookup = ops.Argmax(axis=-1)(Tensor(generate_lookup_tensor(window_size[1]), mstype.float32))
        # print(self.height_lookup, self.width_lookup.shape, self.window_size)
        # self.init_weights()

    # def init_weights(self):
    # pass
    # normal_(self.relative_position_bias_table, std=.02)

    def get_bias2(self):
        # FIXME change to not use one-hot/einsum?
        return reindex_2d_einsum_lookup2(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup
        )

    def get_bias(self):
        # FIXME change to not use one-hot/einsum?
        return reindex_2d_einsum_lookup(
            self.relative_position_bias_table,
            self.window_size[0],
            self.window_size[1],
            self.height_lookup,
            self.width_lookup
        )

    def construct(self, attn):
        return attn + self.get_bias2()


class Attention2d(nn.Cell):
    """ multi-head attention for 2D NCHW tensors"""

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            has_bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = Tensor(dim_head ** -0.5, mstype.float16)

        self.qkv = nn.Conv2d(dim, dim_attn * 3, 1, has_bias=has_bias)
        if rel_pos_cls:
            self.rel_pos = rel_pos_cls(num_heads=self.num_heads)
        else:
            raise NotImplementedError
        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Conv2d(dim_attn, dim_out, 1, has_bias=has_bias)
        self.proj_drop = Dropout(proj_drop)
        self.identity = nn.Identity()
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        B, C, H, W = x.shape

        if self.head_first:
            qkv = self.qkv(x).reshape(B, self.num_heads, self.dim_head * 3, -1)
            q, k, v = ops.unstack(qkv, axis=2)
        else:
            qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.dim_head, -1)
            q, k, v = ops.unstack(qkv, axis=1)

        attn = ops.BatchMatMul(transpose_a=True)(q, k)
        scale = ops.cast(self.scale, attn.dtype)
        attn = attn * scale

        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        else:
            attn = self.identity(attn)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = ops.BatchMatMul(transpose_b=True)(v, attn).reshape(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionCl(nn.Cell):
    """ Channels-last multi-head attention (B, ..., C) """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            dim_head: int = 32,
            has_bias: bool = True,
            expand_first: bool = True,
            head_first: bool = True,
            rel_pos_cls: Callable = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first and dim_out > dim else dim
        assert dim_attn % dim_head == 0, 'attn dim should be divisible by head_dim'
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = Tensor(dim_head ** -0.5, mstype.float32)

        self.qkv = nn.Dense(dim, dim_attn * 3, has_bias=has_bias)
        if rel_pos_cls:
            self.rel_pos = rel_pos_cls(num_heads=self.num_heads)
        else:
            raise NotImplementedError
        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Dense(dim_attn, dim_out, has_bias=has_bias)
        self.proj_drop = Dropout(proj_drop)
        self.identity = nn.Identity()
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):

        B = x.shape[0]
        restore_shape = x.shape[:-1]
        if self.head_first:
            qkv = self.qkv(x).reshape(B, -1, self.num_heads, self.dim_head, 3).transpose(0, 2, 1, 3, 4)
            q, k, v = ops.unstack(qkv, axis=4)
            # q, k, v = ops.split(qkv, output_num=3, axis=4)
        else:
            qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.dim_head).transpose(0, 3, 2, 1, 4)
            # q, k, v = ops.split(qkv, output_num=3, axis=2)
            q, k, v = ops.unstack(qkv, axis=2)

        attn = ops.BatchMatMul(transpose_b=True)(q, k)
        attn = attn * ops.cast(self.scale, attn.dtype)

        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        else:
            attn = self.identity(attn)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = ops.BatchMatMul()(attn, v).transpose(0, 2, 1, 3).reshape(*restore_shape, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = Parameter(init_values * ops.ones(dim, mstype.float32))

    def construct(self, x):
        gamma = self.gamma
        return x * gamma


class LayerScale2d(nn.Cell):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = Parameter(init_values * ops.ones(dim, mstype.float32))

    def construct(self, x):
        gamma = self.gamma.reshape(1, -1, 1, 1)
        return x * gamma


class Downsample2d(nn.Cell):
    """ A downsample pooling cell supporting several maxpool and avgpool modes
    * 'max' - MaxPool2d w/ kernel_size 3, stride 2, padding 1
    * 'max2' - MaxPool2d w/ kernel_size = stride = 2
    * 'avg' - AvgPool2d w/ kernel_size 3, stride 2, padding 1
    * 'avg2' - AvgPool2d w/ kernel_size = stride = 2
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            pool_type: str = 'avg2',
            has_bias: bool = True,
    ):
        super().__init__()
        assert pool_type in ('max', 'max2', 'avg', 'avg2')
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        elif pool_type == 'max2':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode='same')

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, has_bias=has_bias)
        else:
            self.expand = nn.Identity()

    def construct(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


def _init_transformer(cell, name, scheme=''):
    if isinstance(cell, (nn.Conv2d, nn.Dense)):
        if scheme == 'normal':
            normal_(cell.weight, std=.02)
            if cell.bias is not None:
                zeros_(cell.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(cell.weight, std=.02)
            if cell.bias is not None:
                zeros_(cell.bias)
        elif scheme == 'xavier_normal':
            xavier_normal_(cell.weight)
            if cell.bias is not None:
                zeros_(cell.bias)
        else:
            # vit like
            xavier_uniform_(cell.weight)
            if cell.bias is not None:
                if 'mlp' in name:
                    normal_(cell.bias, std=1e-6)
                else:
                    zeros_(cell.bias)


class TransformerBlock2d(nn.Cell):
    """ Transformer block with 2D downsampling
    '2D' NCHW tensor layout

    Some gains can be seen on GPU using a 1D / CL block, BUT w/ the need to switch back/forth to NCHW
    for spatial pooling, the benefit is minimal so ended up using just this variant for CoAt configs.

    This impl was faster on TPU w/ PT XLA than the 1D experiment.
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            stride: int = 1,
            rel_pos_cls: Callable = None,
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(_NORM_MAP[cfg.norm_layer], epsilon=cfg.norm_eps)

        act_layer = _ACT_LAYER_DEFAULT[cfg.act_layer]

        if stride == 2:
            self.shortcut = Downsample2d(dim, dim_out, pool_type=cfg.pool_type, has_bias=cfg.shortcut_bias)
            self.norm1 = nn.SequentialCell(OrderedDict([
                ('norm', norm_layer((dim,))),
                ('down', Downsample2d(dim, dim, pool_type=cfg.pool_type)),
            ]))
        else:
            assert dim == dim_out
            self.shortcut = nn.Identity()
            self.norm1 = norm_layer((dim,))

        self.attn = Attention2d(
            dim,
            dim_out,
            dim_head=cfg.dim_head,
            expand_first=cfg.expand_first,
            has_bias=cfg.attn_bias,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop
        )
        self.ls1 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim_out,))
        self.mlp = ConvMlp(
            in_features=dim_out,
            hidden_features=int(dim_out * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale2d(dim_out, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # def init_weights(self, scheme=''):
    # pass
    # named_apply(partial(_init_transformer, scheme=scheme), self)

    def construct(self, x):
        x = self.shortcut(x) + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def _init_conv(cell, name, scheme=''):
    if isinstance(cell, nn.Conv2d):
        if scheme == 'normal':
            normal_(cell.weight, std=.02)
            if cell.bias is not None:
                zeros_(cell.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(cell.weight, std=.02)
            if cell.bias is not None:
                zeros_(cell.bias)
        elif scheme == 'xavier_normal':
            xavier_normal_(cell.weight)
            if cell.bias is not None:
                zeros_(cell.bias)
        else:
            # efficientnet like
            fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
            fan_out //= cell.groups
            normal_(cell.weight, math.sqrt(2.0 / fan_out))
            if cell.bias is not None:
                zeros_(cell.bias)


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class MbConvBlock(nn.Cell):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: float = 0.
    ):
        super(MbConvBlock, self).__init__()
        mid_chs = make_divisible((out_chs if cfg.expand_output else in_chs) * cfg.expand_ratio, divisor=8)
        groups = num_groups(cfg.group_size, mid_chs)

        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, pool_type=cfg.pool_type, has_bias=cfg.output_bias)
        else:
            self.shortcut = nn.Identity()

        assert cfg.stride_mode in ('pool', '1x1', 'dw')
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if cfg.stride_mode == 'pool':
            # NOTE this is not described in paper, experiment to find faster option that doesn't stride in 1x1
            stride_pool, dilation_2 = stride, dilation[1]
            # FIXME handle dilation of avg pool
        elif cfg.stride_mode == '1x1':
            # NOTE I don't like this option described in paper, 1x1 w/ stride throws info away
            stride_1, dilation_2 = stride, dilation[1]
        else:
            stride_2, dilation_2 = stride, dilation[0]

        if cfg.pre_norm_act:
            self.pre_norm = nn.SequentialCell(
                _NORM_MAP[cfg.norm_layer](in_chs if "layernorm" not in cfg.norm_layer else (in_chs,),
                                          epsilon=cfg.norm_eps),
                _ACT_LAYER_DEFAULT[cfg.act_layer]
            )
        else:
            self.pre_norm = _NORM_MAP[cfg.norm_layer](in_chs, eps=cfg.norm_eps)
        if stride_pool > 1:
            self.down = Downsample2d(in_chs, in_chs, pool_type=cfg.downsample_pool_type)
        else:
            self.down = nn.Identity()
        self.conv1_1x1 = nn.Conv2d(in_chs, mid_chs, 1, stride=stride_1, has_bias=False)
        self.norm1 = nn.SequentialCell(
            _NORM_MAP[cfg.norm_layer](mid_chs if "layernorm" not in cfg.norm_layer else (mid_chs,),
                                      eps=cfg.norm_eps),
            _ACT_LAYER_DEFAULT[cfg.act_layer]()
        )

        if ms.__version__.startswith('1.8'):
            self.conv2_kxk = nn.Conv2d(mid_chs, mid_chs, cfg.kernel_size, stride_2, dilation=dilation_2,
                                       group=groups, has_bias=False)

        attn_kwargs = {}
        if isinstance(cfg.attn_layer, str):
            if cfg.attn_layer == 'se' or cfg.attn_layer == 'eca':
                attn_kwargs['act_layer'] = _ACT_LAYER_DEFAULT[cfg.attn_act_layer]
                attn_kwargs['rd_channels'] = int(cfg.attn_ratio * (out_chs if cfg.expand_output else mid_chs))

        # two different orderings for SE and norm2 (due to some weights and trials using SE before norm2)
        if cfg.attn_early:
            self.se_early = SqueezeExcite(mid_chs, **attn_kwargs)
            self.norm2 = nn.SequentialCell(
                _NORM_MAP[cfg.norm_layer](mid_chs if "layernorm" not in cfg.norm_layer else (mid_chs,),
                                          eps=cfg.norm_eps),
                _ACT_LAYER_DEFAULT[cfg.act_layer]
            )
            self.se = None
        else:
            self.se_early = None
            self.norm2 = nn.SequentialCell(
                _NORM_MAP[cfg.norm_layer](mid_chs if "layernorm" not in cfg.norm_layer else (mid_chs,),
                                          eps=cfg.norm_eps),
                _ACT_LAYER_DEFAULT[cfg.act_layer]()
            )
            self.se = SqueezeExcite(mid_chs, **attn_kwargs)

        self.conv3_1x1 = nn.Conv2d(mid_chs, out_chs, 1, has_bias=cfg.output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # def init_weights(self, scheme=''):
    #     pass
    # named_apply(partial(_init_conv, scheme=scheme), self)

    def construct(self, x):
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)

        # 1x1 expansion conv & norm-act
        x = self.conv1_1x1(x)
        x = self.norm1(x)

        # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut
        return x


def window_partition(x, window_size: List[int]):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.reshape(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, H, W, C)
    return x


def grid_partition(x, grid_size: List[int]):
    B, H, W, C = x.shape
    x = x.reshape(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.reshape(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.transpose(0, 3, 1, 4, 2, 5).reshape(-1, H, W, C)
    return x


def get_rel_pos_cls(cfg: MaxxVitTransformerCfg, window_size):
    rel_pos_cls = None
    if cfg.rel_pos_type == 'bias_tf':
        rel_pos_cls = partial(RelPosBiasTf, window_size=window_size)
    return rel_pos_cls


class PartitionAttentionCl(nn.Cell):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(_NORM_MAP[cfg.norm_layer], epsilon=cfg.norm_eps)
        act_layer = _ACT_LAYER_DEFAULT[cfg.act_layer]

        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer((dim,))
        self.attn = AttentionCl(
            dim,
            dim,
            dim_head=cfg.dim_head,
            has_bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_block:
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = window_reverse(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse(partitioned, self.partition_size, img_size)
        return x

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ParallelPartitionAttention(nn.Cell):
    """ Experimental. Grid and Block partition + single FFN
    NxC tensor layout.
    """

    def __init__(
            self,
            dim: int,
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        assert dim % 2 == 0
        norm_layer = partial(_NORM_MAP[cfg.norm_layer], epsilon=cfg.norm_eps)
        act_layer = _ACT_LAYER_DEFAULT[cfg.act_layer]

        assert cfg.window_size == cfg.grid_size
        self.partition_size = to_2tuple(cfg.window_size)
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer(dim)
        self.attn_block = AttentionCl(
            dim,
            dim // 2,
            dim_head=cfg.dim_head,
            has_bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.attn_grid = AttentionCl(
            dim,
            dim // 2,
            dim_head=cfg.dim_head,
            has_bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]

        partitioned_block = window_partition(x, self.partition_size)
        partitioned_block = self.attn_block(partitioned_block)
        x_window = window_reverse(partitioned_block, self.partition_size, img_size)

        partitioned_grid = grid_partition(x, self.partition_size)
        partitioned_grid = self.attn_grid(partitioned_grid)
        x_grid = grid_reverse(partitioned_grid, self.partition_size, img_size)

        return ops.concat((x_window, x_grid), axis=-1)

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def window_partition_nchw(x, window_size: List[int]):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.transpose(0, 2, 4, 1, 3, 5).reshape(-1, C, window_size[0], window_size[1])
    return windows


def window_reverse_nchw(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[1]
    x = windows.reshape(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
    x = x.transpose(0, 3, 1, 4, 2, 5).reshape(-1, C, H, W)
    return x


def grid_partition_nchw(x, grid_size: List[int]):
    B, C, H, W = x.shape
    x = x.reshape(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    windows = x.transpose(0, 3, 5, 1, 2, 4).reshape(-1, C, grid_size[0], grid_size[1])
    return windows


def grid_reverse_nchw(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    C = windows.shape[1]
    x = windows.reshape(-1, H // grid_size[0], W // grid_size[1], C, grid_size[0], grid_size[1])
    x = x.transpose(0, 3, 4, 1, 5, 2).reshape(-1, C, H, W)
    return x


class PartitionAttention2d(nn.Cell):
    """ Grid or Block partition + Attn + FFN

    '2D' NCHW tensor layout.
    """

    def __init__(
            self,
            dim: int,
            partition_type: str = 'block',
            cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        norm_layer = partial(_NORM_MAP[cfg.norm_layer], epsilon=cfg.norm_eps)

        act_layer = _ACT_LAYER_DEFAULT[cfg.act_layer]

        self.partition_block = partition_type == 'block'
        self.partition_size = to_2tuple(cfg.window_size if self.partition_block else cfg.grid_size)
        rel_pos_cls = get_rel_pos_cls(cfg, self.partition_size)

        self.norm1 = norm_layer((dim,))
        self.attn = Attention2d(
            dim,
            dim,
            dim_head=cfg.dim_head,
            has_bias=cfg.attn_bias,
            head_first=cfg.head_first,
            rel_pos_cls=rel_pos_cls,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
        )
        self.ls1 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=int(dim * cfg.expand_ratio),
            act_layer=act_layer,
            drop=cfg.proj_drop)
        self.ls2 = LayerScale2d(dim, init_values=cfg.init_values) if cfg.init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[-2:]
        if self.partition_block:
            partitioned = window_partition_nchw(x, self.partition_size)
        else:
            partitioned = grid_partition_nchw(x, self.partition_size)

        partitioned = self.attn(partitioned)

        if self.partition_block:
            x = window_reverse_nchw(partitioned, self.partition_size, img_size)
        else:
            x = grid_reverse_nchw(partitioned, self.partition_size, img_size)
        return x

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaxxVitBlock(nn.Cell):
    """ MaxVit conv, window partition + FFN , grid partition + FFN
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            stride: int = 1,
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path: float = 0.,
    ):
        super().__init__()
        self.nchw_attn = transformer_cfg.use_nchw_attn

        conv_cls = MbConvBlock
        self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)

        attn_kwargs = dict(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)
        partition_layer = PartitionAttention2d if self.nchw_attn else PartitionAttentionCl
        self.attn_block = None if transformer_cfg.no_block_attn else partition_layer(**attn_kwargs)
        self.attn_grid = partition_layer(partition_type='grid', **attn_kwargs)

    # def init_weights(self, scheme=''):
    #     pass
    # if self.attn_block is not None:
    #     named_apply(partial(_init_transformer, scheme=scheme), self.attn_block)
    # named_apply(partial(_init_transformer, scheme=scheme), self.attn_grid)
    # named_apply(partial(_init_conv, scheme=scheme), self.conv)

    def construct(self, x):
        # NCHW format
        x = self.conv(x)

        if not self.nchw_attn:
            x = x.transpose(0, 2, 3, 1)  # to NHWC (channels-last)
        if self.attn_block is not None:
            x = self.attn_block(x)
        x = self.attn_grid(x)
        if not self.nchw_attn:
            x = x.transpose(0, 3, 1, 2)  # back to NCHW
        return x


class ParallelMaxxVitBlock(nn.Cell):
    """ MaxVit block with parallel cat(window + grid), one FF
    Experimental timm block.
    """

    def __init__(
            self,
            dim,
            dim_out,
            stride=1,
            num_conv=2,
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            drop_path=0.,
    ):
        super().__init__()

        conv_cls = MbConvBlock
        if num_conv > 1:
            convs = [conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)]
            convs += [conv_cls(dim_out, dim_out, cfg=conv_cfg, drop_path=drop_path)] * (num_conv - 1)
            self.conv = nn.SequentialCell(*convs)
        else:
            self.conv = conv_cls(dim, dim_out, stride=stride, cfg=conv_cfg, drop_path=drop_path)
        self.attn = ParallelPartitionAttention(dim=dim_out, cfg=transformer_cfg, drop_path=drop_path)

    # def init_weights(self, scheme=''):
    #     pass
    # named_apply(partial(_init_transformer, scheme=scheme), self.attn)
    # named_apply(partial(_init_conv, scheme=scheme), self.conv)

    def construct(self, x):
        x = self.conv(x)
        x = x.transpose(0, 2, 3, 1)
        x = self.attn(x)
        x = x.transpose(0, 3, 1, 2)
        return x


def extend_tuple(x, n):
    # pdas a tuple to specified n by padding with last value
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n


class MaxxVitStage(nn.Cell):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 2,
            depth: int = 4,
            feat_size: Tuple[int, int] = (14, 14),
            block_types: Union[str, Tuple[str]] = 'C',
            transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg(),
            conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg(),
            drop_path: Union[float, List[float]] = 0.,
    ):
        super().__init__()
        self.grad_checkpointing = False

        block_types = extend_tuple(block_types, depth)
        blocks = []
        for i, t in enumerate(block_types):
            block_stride = stride if i == 0 else 1
            assert t in ('C', 'T', 'M', 'PM')
            if t == 'C':
                conv_cls = MbConvBlock
                blocks += [conv_cls(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    cfg=conv_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'T':
                rel_pos_cls = get_rel_pos_cls(transformer_cfg, feat_size)
                blocks += [TransformerBlock2d(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    rel_pos_cls=rel_pos_cls,
                    cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'M':
                blocks += [MaxxVitBlock(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    conv_cfg=conv_cfg,
                    transformer_cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            elif t == 'PM':
                blocks += [ParallelMaxxVitBlock(
                    in_chs,
                    out_chs,
                    stride=block_stride,
                    conv_cfg=conv_cfg,
                    transformer_cfg=transformer_cfg,
                    drop_path=drop_path[i],
                )]
            in_chs = out_chs
        self.blocks = nn.SequentialCell(*blocks)

    def construct(self, x):
        x = self.blocks(x)
        return x


class Stem(nn.Cell):

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            has_bias: bool = False,
            act_layer: str = 'gelu',
            norm_layer: str = 'batchnorm2d',
            norm_eps: float = 1e-5,
    ):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)

        self.out_chs = out_chs[-1]
        self.stride = 2

        self.conv1 = nn.Conv2d(in_chs, out_chs[0], kernel_size, stride=2, has_bias=has_bias)
        self.norm1 = nn.SequentialCell(
            _NORM_MAP[norm_layer](out_chs[0] if "layernorm" not in norm_layer else (out_chs[0],),
                                  eps=norm_eps),
            _ACT_LAYER_DEFAULT[act_layer]()
        )
        self.conv2 = nn.Conv2d(out_chs[0], out_chs[1], kernel_size, stride=1, has_bias=has_bias)

    # def init_weights(self, scheme=''):
    #     pass
    # named_apply(partial(_init_conv, scheme=scheme), self)

    def construct(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


def cfg_window_size(cfg: MaxxVitTransformerCfg, img_size: Tuple[int, int]):
    if cfg.window_size is not None:
        assert cfg.grid_size
        return cfg
    partition_size = img_size[0] // cfg.partition_ratio, img_size[1] // cfg.partition_ratio
    cfg = replace(cfg, window_size=partition_size, grid_size=partition_size)
    return cfg


class NormMlpHead(nn.Cell):

    def __init__(
            self,
            in_features,
            num_classes,
            norm_layer,
            hidden_size=None,
            pool_type='avg',
            drop_rate=0.,
            act_layer='tanh',
    ):
        super().__init__()

        self.drop_rate = drop_rate
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        self.use_conv = not pool_type

        act_layer = _ACT_LAYER_DEFAULT[act_layer]
        linear_layer = partial(nn.Conv2d, kernel_size=1) if self.use_conv else nn.Dense

        self.global_pool = GlobalAvgPooling(keep_dims=False)
        self.norm = norm_layer((in_features,))
        self.flatten = nn.Flatten() if pool_type else nn.Identity()
        if hidden_size:
            self.pre_logits = nn.SequentialCell(OrderedDict([
                ('fc', linear_layer(in_features, hidden_size)),
                ('act', act_layer()),
            ]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = Dropout(self.drop_rate)
        self.fc = linear_layer(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def construct(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


def _overlay_kwargs(cfg: MaxxVitCfg, **kwargs):
    transformer_kwargs = {}
    conv_kwargs = {}
    base_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith('transformer_'):
            transformer_kwargs[k.replace('transformer_', '')] = v
        elif k.startswith('conv_'):
            conv_kwargs[k.replace('conv_', '')] = v
        else:
            base_kwargs[k] = v
    cfg = replace(
        cfg,
        transformer_cfg=replace(cfg.transformer_cfg, **transformer_kwargs),
        conv_cfg=replace(cfg.conv_cfg, **conv_kwargs),
        **base_kwargs
    )
    return cfg


class MaxxVit(nn.Cell):
    """ CoaTNet + MaxVit base model.

    Highly configurable for different block compositions, tensor layouts, pooling types.
    """

    def __init__(
            self,
            cfg: MaxxVitCfg,
            img_size: Union[int, Tuple[int, int]] = 224,
            in_channels: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        if kwargs:
            cfg = _overlay_kwargs(cfg, **kwargs)
        transformer_cfg = cfg_window_size(cfg.transformer_cfg, img_size)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = cfg.embed_dim[-1]
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        self.stem = Stem(
            in_chs=in_channels,
            out_chs=cfg.stem_width,
            has_bias=cfg.stem_bias,
            act_layer=cfg.conv_cfg.act_layer,
            norm_layer=cfg.conv_cfg.norm_layer,
            norm_eps=cfg.conv_cfg.norm_eps,
        )
        stride = self.stem.stride
        self.feature_info += [dict(num_chs=self.stem.out_chs, reduction=2, cell='stem')]
        feat_size = tuple([i // s for i, s in zip(img_size, to_2tuple(stride))])

        num_stages = len(cfg.embed_dim)
        assert len(cfg.depths) == num_stages
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(cfg.depths))]
        in_chs = self.stem.out_chs
        stages = []
        cur = 0
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg.embed_dim[i]
            feat_size = tuple([(r - 1) // stage_stride + 1 for r in feat_size])
            stages += [MaxxVitStage(
                in_chs,
                out_chs,
                depth=cfg.depths[i],
                block_types=cfg.block_type[i],
                conv_cfg=cfg.conv_cfg,
                transformer_cfg=transformer_cfg,
                feat_size=feat_size,
                drop_path=dpr[cur:cur + cfg.depths[i]],
            )]
            stride *= stage_stride
            in_chs = out_chs
            cur += cfg.depths[i]
            self.feature_info += [dict(num_chs=out_chs, reduction=stride, cell=f'stages.{i}')]
        self.stages = nn.SequentialCell(*stages)

        final_norm_layer = partial(_NORM_MAP[cfg.transformer_cfg.norm_layer], epsilon=cfg.transformer_cfg.norm_eps)
        self.head_hidden_size = cfg.head_hidden_size
        assert self.head_hidden_size
        self.norm = nn.Identity()
        self.head = NormMlpHead(
            self.num_features,
            num_classes,
            hidden_size=self.head_hidden_size,
            pool_type=global_pool,
            drop_rate=drop_rate,
            norm_layer=final_norm_layer,
        )

        # Weight init (default PyTorch init works well for AdamW if scheme not set)
        assert cfg.weight_init in ('', 'normal', 'trunc_normal', 'xavier_normal', 'vit_eff')
        # if cfg.weight_init:
        #     pass
        # named_apply(partial(self._init_weights, scheme=cfg.weight_init), self)

    # def _init_weights(self, cell, name, scheme=''):
    #     if hasattr(cell, 'init_weights'):
    #         try:
    #             cell.init_weights(scheme=scheme)
    #         except TypeError:
    #             cell.init_weights()

    def no_weight_decay(self):
        return {
            k for k, _ in self.named_parameters()
            if any(n in k for n in ["relative_position_bias_table", "rel_pos.mlp"])}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _tf_cfg():
    return dict(
        conv_cfg=MaxxVitConvCfg(
            norm_eps=1e-3,
            act_layer='gelu_tanh',
            padding='same',
        ),
        transformer_cfg=MaxxVitTransformerCfg(
            norm_eps=1e-5,
            act_layer='gelu_tanh',
            head_first=False,  # heads are interleaved (q_nh, q_hdim, k_nh, q_hdim, ....)
            rel_pos_type='bias_tf',
        ),
    )


model_cfgs = dict(
    # Trying to be like the MaxViT paper configs
    maxvit_tiny_tf=MaxxVitCfg(
        embed_dim=(64, 128, 256, 512),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=512,
        **_tf_cfg(),
    ),
    maxvit_small_tf=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 5, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=768,
        **_tf_cfg(),
    ),
    maxvit_base_tf=MaxxVitCfg(
        embed_dim=(96, 192, 384, 768),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=64,
        stem_bias=True,
        head_hidden_size=768,
        **_tf_cfg(),
    ),
    maxvit_large_tf=MaxxVitCfg(
        embed_dim=(128, 256, 512, 1024),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=128,
        stem_bias=True,
        head_hidden_size=1024,
        **_tf_cfg(),
    ),
    maxvit_xlarge_tf=MaxxVitCfg(
        embed_dim=(192, 384, 768, 1536),
        depths=(2, 6, 14, 2),
        block_type=('M',) * 4,
        stem_width=192,
        stem_bias=True,
        head_hidden_size=1536,
        **_tf_cfg(),
    ),
)


def _create_maxxvit(pretrained, variant, cfg_variant=None, **kwargs):
    if cfg_variant is None:
        if variant in model_cfgs:
            cfg_variant = variant
        else:
            cfg_variant = '_'.join(variant.split('_')[:-1])

    model_cfg = model_cfgs[cfg_variant]
    default_cfg = default_cfgs[variant]

    model_maxvit = MaxxVit(model_cfg, **kwargs)

    if pretrained:
        load_pretrained(model_maxvit, default_cfg, kwargs.get("num_classes", 1000), kwargs.get("in_channels", 3))

    return model_maxvit


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.conv1', 'classifier': 'head.fc',
        'fixed_input_size': True,
        **kwargs
    }


default_cfgs = {
    # MaxViT models ported from official Tensorflow impl
    'maxvit_tiny_tf_224': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_tiny_tf_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_tiny_tf_512': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_small_tf_224': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_small_tf_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_small_tf_512': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_base_tf_224': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_base_tf_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_base_tf_512': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash'),
    'maxvit_large_tf_224': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'maxvit_large_tf_384': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0, crop_mode='squash'),
    'maxvit_large_tf_512': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 512, 512), pool_size=(16, 16), crop_pct=1.0, crop_mode='squash')
}


@register_model
def maxvit_tiny_tf_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=224, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_tiny_tf_224', 'maxvit_tiny_tf', **model_args)


@register_model
def maxvit_tiny_tf_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=384, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_tiny_tf_384', 'maxvit_tiny_tf', **model_args)


@register_model
def maxvit_tiny_tf_512(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=512, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_tiny_tf_512', 'maxvit_tiny_tf', **model_args)


@register_model
def maxvit_small_tf_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=224, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_small_tf_224', 'maxvit_small_tf', **model_args)


@register_model
def maxvit_small_tf_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=384, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_small_tf_384', 'maxvit_small_tf', **model_args)


@register_model
def maxvit_small_tf_512(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=512, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_small_tf_512', 'maxvit_small_tf', **model_args)


@register_model
def maxvit_base_tf_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=224, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_base_tf_224', 'maxvit_base_tf', **model_args)


@register_model
def maxvit_base_tf_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=384, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_base_tf_384', 'maxvit_base_tf', **model_args)


@register_model
def maxvit_base_tf_512(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=512, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_base_tf_512', 'maxvit_base_tf', **model_args)


@register_model
def maxvit_large_tf_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=224, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_large_tf_224', 'maxvit_large_tf', **model_args)


@register_model
def maxvit_large_tf_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=384, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_large_tf_384', 'maxvit_large_tf', **model_args)


@register_model
def maxvit_large_tf_512(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(in_channels=in_channels, num_classes=num_classes, img_size=512, **kwargs)
    return _create_maxxvit(pretrained, 'maxvit_large_tf_512', 'maxvit_large_tf', **model_args)


if __name__ == "__main__":
    from mindspore import set_seed

    set_seed(1234)
    model = maxvit_tiny_tf_224()
    data = Tensor(np.random.randn(1, 3, 224, 224), mstype.float32)
    out = model(data)
    print(out.sum())

    # print(sum([p.size for p in model.trainable_params()]))
    # total = 0
    # for p in model.trainable_params():
    #     # if "stages.0" in p.name:
    #     total += p.size
    #     print(p.name)
    # print(total)
