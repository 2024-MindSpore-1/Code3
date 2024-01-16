import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, Tensor, Parameter
from mindspore import ops

from mindcv.models.dwconv import DeepWiseConv2D
from mindcv.models.layers.drop_path import DropPath, Dropout
from mindcv.models.layers.pooling import GlobalAvgPooling
from mindcv.models.layers.squeeze_excite import SqueezeExcite
from mindcv.models.layers.weight_init import trunc_normal_, zeros_, ones_
from mindcv.models.registry import register_model

__all__ = [
    "maxvit_tiny_pytorch",
]


def _get_conv_output_shape(input_size: Tuple[int, int], kernel_size: int, stride: int, padding: int) -> Tuple[int, int]:
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


def _make_block_input_shapes(input_size: Tuple[int, int], n_blocks: int) -> List[Tuple[int, int]]:
    """Util function to check that the input size is correct for a MaxVit configuration."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_relative_position_index(height: int, width: int) -> Tensor:
    coords = np.stack(np.meshgrid(np.arange(height), np.arange(width)), axis=0)
    coords_flat = np.reshape(coords, (coords.shape[0], -1))
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.transpose(1, 2, 0)
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class MBConv(nn.Cell):
    """MBConv: Mobile Inverted Residual Bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Cell]): Activation function.
        norm_layer (Callable[..., nn.Cell]): Normalization function.
        p_stochastic_dropout (float): Probability of stochastic depth.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_ratio: float,
            squeeze_ratio: float,
            stride: int,
            activation_layer: Callable[..., nn.Cell],
            norm_layer: Callable[..., nn.Cell],
            p_stochastic_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        proj: Sequence[nn.Cell]
        self.proj: nn.Cell

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, has_bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, pad_mode='same')] + proj  # type: ignore
            self.proj = nn.SequentialCell(*proj)
        else:
            self.proj = nn.Identity()  # type: ignore

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.drop_path = DropPath(p_stochastic_dropout)  # type: ignore
        else:
            self.drop_path = nn.Identity()  # type: ignore

        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        _layers["conv_a"] = nn.SequentialCell(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, has_bias=norm_layer is None),
            norm_layer(mid_channels),
            activation_layer()
        )
        _layers['conv_b'] = nn.SequentialCell(
            DeepWiseConv2D(mid_channels, mid_channels, kernel_size=3, stride=stride, group=mid_channels,
                           has_bias=norm_layer is None),
            norm_layer(mid_channels),
            activation_layer()
        )
        _layers["squeeze_excitation"] = SqueezeExcite(mid_channels, rd_channels=sqz_channels, act_layer=nn.SiLU)
        _layers["conv_c"] = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, has_bias=True)

        self.layers = nn.SequentialCell(_layers)

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H / stride, W / stride].
        """
        res = self.proj(x)
        x = self.drop_path(self.layers(x))
        return res + x


class RelativePositionalMultiHeadAttention(nn.Cell):
    """Relative Positional Multi-Head Attention.

    Args:
        feat_dim (int): Number of input features.
        head_dim (int): Number of features per head.
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(
            self,
            feat_dim: int,
            head_dim: int,
            max_seq_len: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Dense(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim ** -0.5

        self.merge = nn.Dense(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = Parameter(
            Tensor(np.zeros(((2 * self.size - 1) * (2 * self.size - 1), self.n_heads)), dtype=mstype.float32))

        relative_position_index = Tensor(_get_relative_position_index(self.size, self.size), mstype.int32).reshape(-1)
        # relative_position_index = ops.OneHot()(relative_position_index, )
        # print(_get_relative_position_index(self.size, self.size).max(), self.size)
        # print(relative_position_index.shape)
        self.relative_position_index = relative_position_index
        # initialize with truncated normal the bias
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(axis=-1)

    def get_relative_positional_bias(self) -> Tensor:
        bias_index = self.relative_position_index  # type: ignore
        # relative_bias = ops.gather(self.relative_position_bias_table, bias_index, axis=0)
        # relative_bias = self.relative_position_bias_table[bias_index]
        relative_bias = ops.Gather()(self.relative_position_bias_table, bias_index, 0)
        relative_bias = relative_bias.reshape(self.max_seq_len, self.max_seq_len, -1)  # type: ignore
        relative_bias = relative_bias.transpose(2, 0, 1)
        return relative_bias.expand_dims(0)

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, G, P, D].
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = ops.split(qkv, axis=-1, output_num=3)

        q = q.reshape(B, G, P, H, DH).transpose(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).transpose(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).transpose(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = ops.BatchMatMul(transpose_b=True)(q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = self.softmax(dot_prod + pos_bias)

        out = ops.BatchMatMul()(dot_prod, v)
        out = out.transpose(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Cell):
    """Permute the axes of a tensor."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def construct(self, x: Tensor) -> Tensor:
        res = x.swapaxes(self.a, self.b)
        return res


class WindowPartition(nn.Cell):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def construct(self, x: Tensor, p: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P, W/P, P*P, C].
        """
        B, C, H, W = x.shape
        P = p
        # chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.transpose(0, 2, 4, 3, 5, 1)
        # colapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x


class WindowDepartition(nn.Cell):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def construct(self, x: Tensor, p: int, h_partitions: int, w_partitions: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.transpose(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)
        return x


class PartitionAttentionLayer(nn.Cell):
    """
    Layer for partitioning the input tensor into non-overlapping windows and applying attention to each window.

    Args:
        in_channels (int): Number of input channels.
        head_dim (int): Dimension of each attention head.
        partition_size (int): Size of the partitions.
        partition_type (str): Type of partitioning to use. Can be either "grid" or "window".
        grid_size (Tuple[int, int]): Size of the grid to partition the input tensor into.
        mlp_ratio (int): Ratio of the  feature size expansion in the MLP layer.
        activation_layer (Callable[..., nn.Cell]): Activation function to use.
        norm_layer (Callable[..., nn.Cell]): Normalization function to use.
        attention_dropout (float): Dropout probability for the attention layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        p_stochastic_dropout (float): Probability of dropping out a partition.
    """

    def __init__(
            self,
            in_channels: int,
            head_dim: int,
            # partitioning parameters
            partition_size: int,
            partition_type: str,
            # grid size needs to be known at initialization time
            # because we need to know hamy relative offsets there are in the grid
            grid_size: Tuple[int, int],
            mlp_ratio: int,
            activation_layer: Callable[..., nn.Cell],
            norm_layer: Callable[..., nn.Cell],
            attention_dropout: float,
            mlp_dropout: float,
            p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()

        self.attn_layer = nn.SequentialCell(
            norm_layer((in_channels,)),
            # it's always going to be partition_size ** 2 because
            # of the axis swap in the case of grid partitioning
            RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size ** 2),
            Dropout(attention_dropout),
        )

        # pre-normalization similar to transformer layers
        self.mlp_layer = nn.SequentialCell(
            nn.LayerNorm((in_channels,)),
            nn.Dense(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Dense(in_channels * mlp_ratio, in_channels),
            Dropout(mlp_dropout),
        )

        # layer scale factors
        self.drop_path = DropPath(p_stochastic_dropout)

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """

        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
        gh, gw = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.drop_path(self.attn_layer(x))
        x = x + self.drop_path(self.mlp_layer(x))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, gh, gw)

        return x


class MaxVitLayer(nn.Cell):
    """
    MaxVit layer consisting of a MBConv layer followed by a PartitionAttentionLayer with `window` and a PartitionAttentionLayer with `grid`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Cell]): Activation function.
        norm_layer (Callable[..., nn.Cell]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        grid_size (Tuple[int, int]): Size of the input feature grid.
    """

    def __init__(
            self,
            # conv parameters
            in_channels: int,
            out_channels: int,
            squeeze_ratio: float,
            expansion_ratio: float,
            stride: int,
            # conv + transformer parameters
            norm_layer: Callable[..., nn.Cell],
            activation_layer: Callable[..., nn.Cell],
            # transformer parameters
            head_dim: int,
            mlp_ratio: int,
            mlp_dropout: float,
            attention_dropout: float,
            p_stochastic_dropout: float,
            # partitioning parameters
            partition_size: int,
            grid_size: Tuple[int, int],
    ) -> None:
        super().__init__()

        layers: OrderedDict = OrderedDict()

        # convolutional layer
        layers["MBconv"] = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        # attention layers, block -> grid
        layers["window_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        layers["grid_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        x = self.layers(x)
        return x


class MaxVitBlock(nn.Cell):
    """
    A MaxVit block consisting of `n_layers` MaxVit layers.

     Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        activation_layer (Callable[..., nn.Cell]): Activation function.
        norm_layer (Callable[..., nn.Cell]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        input_grid_size (Tuple[int, int]): Size of the input feature grid.
        n_layers (int): Number of layers in the block.
        p_stochastic (List[float]): List of probabilities for stochastic depth for each layer.
    """

    def __init__(
            self,
            # conv parameters
            in_channels: int,
            out_channels: int,
            squeeze_ratio: float,
            expansion_ratio: float,
            # conv + transformer parameters
            norm_layer: Callable[..., nn.Cell],
            activation_layer: Callable[..., nn.Cell],
            # transformer parameters
            head_dim: int,
            mlp_ratio: int,
            mlp_dropout: float,
            attention_dropout: float,
            # partitioning parameters
            partition_size: int,
            input_grid_size: Tuple[int, int],
            # number of layers
            n_layers: int,
            p_stochastic: List[float],
    ) -> None:
        super().__init__()
        if not len(p_stochastic) == n_layers:
            raise ValueError(f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}.")

        self.layers = nn.CellList()
        # account for the first stride of the first layer
        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=3, stride=2, padding=1)

        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers += [
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                ),
            ]

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MaxVit(nn.Cell):
    """
    Implements MaxVit Transformer from the `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_ paper.
    Args:
        input_size (Tuple[int, int]): Size of the input image.
        stem_channels (int): Number of channels in the stem.
        partition_size (int): Size of the partitions.
        block_channels (List[int]): Number of channels in each block.
        block_layers (List[int]): Number of layers in each block.
        drop_path_rate (float): Probability of stochastic depth. Expands to a list of probabilities for each layer that scales linearly to the specified value.
        squeeze_ratio (float): Squeeze ratio in the SE Layer. Default: 0.25.
        expansion_ratio (float): Expansion ratio in the MBConv bottleneck. Default: 4.
        norm_layer (Callable[..., nn.Cell]): Normalization function. Default: None (setting to None will produce a `BatchNorm2d(eps=1e-3, momentum=0.99)`).
        activation_layer (Callable[..., nn.Cell]): Activation function Default: nn.GELU.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Expansion ratio of the MLP layer. Default: 4.
        mlp_dropout (float): Dropout probability for the MLP layer. Default: 0.0.
        attention_dropout (float): Dropout probability for the attention layer. Default: 0.0.
        num_classes (int): Number of classes. Default: 1000.
    """

    def __init__(
            self,
            # input size parameters
            input_size: Tuple[int, int],
            # stem and task parameters
            stem_channels: int,
            # partitioning parameters
            partition_size: int,
            # block parameters
            block_channels: List[int],
            block_layers: List[int],
            # attention head dimensions
            head_dim: int,
            drop_path_rate: float,
            # conv + transformer parameters
            # norm_layer is applied only to the conv layers
            # activation_layer is applied both to conv and transformer layers
            norm_layer: Optional[Callable[..., nn.Cell]] = None,
            activation_layer: Callable[..., nn.Cell] = nn.GELU,
            # conv parameters
            squeeze_ratio: float = 0.25,
            expansion_ratio: float = 4,
            # transformer parameters
            mlp_ratio: int = 4,
            mlp_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            # task parameters
            num_classes: int = 1000,
            **kwargs
    ) -> None:
        super().__init__()

        input_channels = 3

        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L1029-L1030
        # for the exact parameters used in batchnorm
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99)

        # Make sure input size will be divisible by the partition size in all blocks
        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                raise ValueError(
                    f"Input size {block_input_size} of block {idx} is not divisible by partition size {partition_size}. "
                    f"Consider changing the partition size or the input size.\n"
                    f"Current configuration yields the following block input sizes: {block_input_sizes}."
                )

        # stem
        self.stem = nn.SequentialCell(
            nn.Conv2d(input_channels, stem_channels, kernel_size=3, stride=2),
            norm_layer(stem_channels), activation_layer(),
            nn.Conv2d(stem_channels, stem_channels, 3, has_bias=True)
        )

        # account for stem stride
        input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        self.partition_size = partition_size

        # blocks
        self.blocks = nn.CellList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        # precompute the stochastich depth probabilities from 0 to drop_path_rate
        # since we have N blocks with L layers, we will have N * L probabilities uniformly distributed
        # over the range [0, drop_path_rate]
        p_stochastic = np.linspace(0, drop_path_rate, sum(block_layers)).tolist()

        p_idx = 0
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels, block_layers):
            self.blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx: p_idx + num_layers],
                ),
            )
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers

        # see https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L1137-L1158
        # for why there is Linear -> Tanh -> Linear
        self.classifier = nn.SequentialCell(
            GlobalAvgPooling(keep_dims=False),
            nn.LayerNorm((block_channels[-1],)),
            nn.Dense(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Dense(block_channels[-1], num_classes, has_bias=False),
        )

        self.init_weights()

    def construct(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for name, cell in self.cells_and_names():
            self._init_vit_weights(cell, name)

    def _init_vit_weights(self, cell: nn.Cell, name: str = '', head_bias: float = 0.):
        """ ViT weight initialization
        * When called without n, head_bias, jax_impl args it will behave exactly the same
          as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
        * When called w/ valid n (cell name) and jax_impl=True, will (hopefully) match JAX impl
        """
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            # NOTE conv was left to pytorch default in my original init
            trunc_normal_(cell.weight, std=0.02)
            if cell.bias is not None:
                zeros_(cell.bias)
        elif isinstance(cell, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            ones_(cell.gamma)
            zeros_(cell.beta)


def _maxvit(
        # stem parameters
        stem_channels: int,
        # block parameters
        block_channels: List[int],
        block_layers: List[int],
        drop_path_rate: float,
        # partitioning parameters
        partition_size: int,
        # transformer parameters
        head_dim: int,
        # Weights API
        weights=None,
        progress: bool = False,
        # kwargs,
        **kwargs: Any,
) -> MaxVit:
    input_size = kwargs.pop("input_size", (224, 224))

    model = MaxVit(
        stem_channels=stem_channels,
        block_channels=block_channels,
        block_layers=block_layers,
        drop_path_rate=drop_path_rate,
        head_dim=head_dim,
        partition_size=partition_size,
        input_size=input_size,
        **kwargs,
    )

    return model


@register_model
def maxvit_tiny_pytorch(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """
    Constructs a maxvit_t architecture from
    `MaxViT: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_.

    Args:
        weights (:class:`~torchvision.models.MaxVit_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MaxVit_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.maxvit.MaxVit``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/maxvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MaxVit_T_Weights
        :members:
    """

    return _maxvit(
        stem_channels=64,
        block_channels=[64, 128, 256, 512],
        block_layers=[2, 2, 5, 2],
        head_dim=32,
        partition_size=7,
        num_classes=num_classes,
        **kwargs,
    )


if __name__ == "__main__":
    from mindspore import set_seed

    set_seed(2345)
    model = maxvit_tiny_pytorch(drop_path_rate=0.2)
    data = Tensor(np.random.randn(1, 3, 224, 224), mstype.float32)
    out = model(data)
    print(sum([p.size for p in model.trainable_params()]))
    print(out.sum())
