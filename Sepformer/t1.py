import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from typing import Optional
from mindspore import Parameter

from src.normalization import LayerNorm
from src.attention import MultiheadAttention, PositionalwiseFeedForward
# from normalization import LayerNorm
# from attention import MultiheadAttention, PositionalwiseFeedForward
from collections import OrderedDict as OD

class PositionalEncoding(nn.Cell):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        # pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        zeros = ops.Zeros()
        pe = zeros((self.max_len, input_size), mindspore.float32)
        # positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        positions = np.arange(start=0, stop=self.max_len, dtype=mindspore.float32)
        expand_dims = ops.ExpandDims()
        positions = expand_dims(positions, 1)
        # denominator = torch.exp(
        #     torch.arange(0, input_size, 2).float()
        #     * -(math.log(10000.0) / input_size)
        # )
        denominator = np.arange(0, input_size, 2, mindspore.float32) * -(math.log(10000.0) / input_size)
        exp = ops.Exp()
        denominator = exp(denominator)
        sin = ops.Sin()
        cos = ops.Cos()
        pe[:, 0::2] = sin(positions * denominator)
        pe[:, 1::2] = cos(positions * denominator)
        pe = expand_dims(pe, 0)
        # self.register_buffer("pe", pe)
        self.pe = Parameter(pe, name="pe", requires_grad=False)

    def construct(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        # return self.pe[:, : x.size(1)].clone().detach()

        # ?
        # return Parameter.clone(self.pe[:, : x.shape[1]])
        return self.pe[:, : x.shape[1]]

class TransformerEncoderLayer(nn.Cell):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=False,
        k=0
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                # kdim=kdim,
                # vdim=vdim,
            )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(1.0-dropout)
        self.dropout2 = nn.Dropout(1.0-dropout)

        self.normalize_before = normalize_before

    def construct(
        self,
        src,
        src_mask: Optional[mindspore.Tensor] = None,
        src_key_padding_mask: Optional[mindspore.Tensor] = None,
        pos_embs: Optional[mindspore.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            # key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output, self_attn


class TransformerEncoder(nn.Cell):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
    ):
        super().__init__()
        self.layers = nn.CellList(
            [TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type
             ) for i in range(num_layers)
             ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def construct(
        self,
        src,
        src_mask: Optional[mindspore.Tensor] = None,
        src_key_padding_mask: Optional[mindspore.Tensor] = None,
        pos_embs: Optional[mindspore.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []

        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


if __name__ == '__main__':
    from mindspore import context
    target = "Ascend"
    device_id = 6

    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    context.set_context(device_id=device_id)
    shape = (8, 60, 512)
    uniformreal = ops.UniformReal(seed=2)
    x = uniformreal(shape)
    net = TransformerEncoder(2, 8, 512, d_model=512)
    output, _ = net(x)
    print(output.shape)


