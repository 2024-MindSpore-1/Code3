import copy
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.Transformer import TransformerEncoder, PositionalEncoding
from src.linear import Linear


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    # if norm == "gln":
    #     return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    # if norm == "cln":
    #     return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Cell):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            group=1,
            has_bias=False,
            pad_mode='pad',
            padding=0,
        )
        self.in_channels = in_channels
        self.expand_dims = ops.ExpandDims()
        self.relu = ops.ReLU()

    def construct(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = self.expand_dims(x, 1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = self.relu(x)

        return x


class Decoder(nn.Conv1dTranspose):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.expand_dims = ops.ExpandDims()
        # self.squeeze = ops.Squeeze()

    def construct(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : mindspore.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """
        # print("x.ndim(), shape", x.ndim(), x.shape)
        # if x.ndim() not in [2, 3]:
        #     raise RuntimeError(
        #         "{} accept 3/4D tensor as input".format(self.__name__)
        #     )
        # print("x before = ", x.shape)
        x = super().construct(x if x.ndim == 3 else self.expand_dims(x, 1))
        # print("x after = ", x.shape)
        # if x.ndim == 3:
        #     x = self.expand_dims(x, 1)

        if ops.squeeze(x).ndim == 1:
            x = ops.squeeze(x, 1)
        else:
            x = ops.squeeze(x)
        return x


class SBTransformerBlock(nn.Cell):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
            self,
            num_layers,
            d_model,
            nhead,
            d_ffn=2048,
            input_shape=None,
            kdim=None,
            vdim=None,
            dropout=0.1,
            activation="relu",
            use_positional_encoding=False,
            norm_before=False,
            attention_type="regularMHA",
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model)

    def construct(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            return self.mdl(x + pos_enc)[0]
        else:
            return self.mdl(x)[0]


class Dual_Computation_Block(nn.Cell):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
            self,
            intra_mdl,
            inter_mdl,
            out_channels,
            norm="ln",
            skip_around_intra=True,
            linear_layer_after_inter_intra=True,  # False
    ):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            # if isinstance(intra_mdl, SBRNNBlock):
            # self.intra_linear = Linear(
            # out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
            # )
            # else:
            self.intra_linear = Linear(
                out_channels, input_size=out_channels
            )

            # if isinstance(inter_mdl, SBRNNBlock):
            # self.inter_linear = Linear(
            # out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
            # )
            # else:
            self.inter_linear = Linear(
                out_channels, input_size=out_channels
            )
        self.transpose = ops.Transpose()

    def construct(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = self.transpose(x, (0, 3, 2, 1)).copy().view((B * S, K, N))
        # [BS, K, H]

        intra = self.intra_mdl(intra)

        # [BS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = self.transpose(intra, (0, 3, 2, 1)).copy()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = self.transpose(intra, (0, 2, 3, 1)).copy().view((B * K, S, N))
        # [BK, S, H]
        inter = self.inter_mdl(inter)

        # [BK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        # [B, K, S, N]
        inter = inter.view((B, K, S, N))
        # [B, N, K, S]
        inter = self.transpose(inter, (0, 3, 1, 2)).copy()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class Dual_Path_Model(nn.Cell):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            intra_model,
            inter_model,
            num_layers=1,
            norm="ln",
            K=200,
            num_spks=2,
            skip_around_intra=True,
            linear_layer_after_inter_intra=True,  # False
            use_global_pos_enc=False,
            max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, pad_mode='pad', padding=0, has_bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.CellList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1, pad_mode='valid', has_bias=True
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, kernel_size=1, pad_mode='valid', has_bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.SequentialCell(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, pad_mode='valid', has_bias=True), nn.Tanh()
        )
        self.output_gate = nn.SequentialCell(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, pad_mode='valid', has_bias=True), nn.Sigmoid()
        )
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze(-1)
        self.zeros = ops.Zeros()

    def construct(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.expand_dims(x, -1)
        x = self.norm(x)
        x = self.squeeze(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose([0, 2, 1])).transpose([0, 2, 1]) + x * (
                    x.shape[1] ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view((B * self.num_spks, -1, K, S))

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view((B, self.num_spks, N, L))
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose([1, 0, 2, 3])

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        # shape = (B, N, gap)
        concat_op = ops.Concat(axis=2)
        if gap > 0:
            # pad = mindspore.Tensor(np.zeros((B, N, gap)), dtype=input.dtype)
            pad = self.zeros((B, N, gap), input.dtype)
            input = concat_op([input, pad])

        # _pad = mindspore.Tensor(np.zeros((B, N, P)), dtype=input.dtype)
        _pad = self.zeros((B, N, P), input.dtype)
        input = concat_op([_pad, input, _pad])

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].copy().view((B, N, -1, K))
        input2 = input[:, :, P:].copy().view((B, N, -1, K))

        concat_op = ops.Concat(axis=3)
        input = (
            concat_op([input1, input2]).view((B, N, -1, K)).transpose([0, 1, 3, 2])
        )

        return input.copy(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose([0, 1, 3, 2]).copy().view((B, N, -1, K * 2))

        input1 = input[:, :, :, :K].copy().view((B, N, -1))[:, :, P:]
        input2 = input[:, :, :, K:].copy().view((B, N, -1))[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input
