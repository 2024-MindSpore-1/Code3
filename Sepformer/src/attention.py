"""Library implementing attention modules.
"""

import os
import sys
import logging
from typing import Optional
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore import Tensor



logger = logging.getLogger(__name__)

class MultiheadAttention(nn.Cell):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """
    #
    # def __init__(
    #     self,
    #     nhead,
    #     d_model,
    #     dropout=0.0,
    #     add_bias_kv=False,
    #     add_zero_attn=False,
    #     kdim=None,
    #     vdim=None,
    # ):
    def __init__(
            self,
            # batch_size,
            # src_seq_length,
            # tgt_seq_length,
            nhead,
            d_model,
            dropout=0.0
    ):
        super().__init__()

        # self.att = nn.MultiheadAttention(
        #     embed_dim=d_model,
        #     num_heads=nhead,
        #     dropout=dropout,
        #     bias=bias,
        #     add_bias_kv=add_bias_kv,
        #     add_zero_attn=add_zero_attn,
        #     kdim=kdim,
        #     vdim=vdim,
        # )

        self.nhead = nhead
        self.d_model = d_model
        self.dropout = float(dropout)
        # self.att = None

        # self.att = nn.transformer.MultiHeadAttention(
        #     batch_size=batch_size,
        #     src_seq_length=src_seq_length,
        #     tgt_seq_length=tgt_seq_length,
        #     hidden_size=d_model,
        #     num_heads=nhead,
        #     hidden_dropout_rate=dropout,
        #     attention_dropout_rate=dropout,
        # )

        self.ones = ops.Ones()

    def construct(
        self,
        query,
        key,
        value,
        # attn_mask,
        attn_mask: Optional[mindspore.Tensor] = None,
        # key_padding_mask: Optional[mindspore.Tensor] = None,
        # return_attn_weights: Optional[mindspore.Tensor] = True,
        pos_embs: Optional[mindspore.Tensor] = None,
    ):
        """
        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        pos_embs: torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # give tensors of shape (time, batch, fea)
        # query = query.permute(1, 0, 2)
        # key = key.permute(1, 0, 2)
        # value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask

        batch_size, src_seq_length, _ = query.shape
        _, tgt_seq_length, _ = key.shape

        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs
        else:
            # attn_mask = Tensor(np.ones((batch_size, src_seq_length, tgt_seq_length)), mstype.float16)
            attn_mask = self.ones((batch_size, src_seq_length, tgt_seq_length), mstype.float16)
        att = nn.transformer.MultiHeadAttention(
            batch_size=batch_size,
            src_seq_length=src_seq_length,
            tgt_seq_length=tgt_seq_length,
            hidden_size=self.d_model,
            num_heads=self.nhead,
            hidden_dropout_rate=self.dropout,
            attention_dropout_rate=self.dropout,
        )

        output, attention_weights = att(
            query,
            key,
            value,
            attention_mask=attn_mask,
        )

        return output, attention_weights

        # if return_attn_weights:
        #     output, attention_weights = output
        #     # reshape the output back to (batch, time, fea)
        #     output = output.permute(1, 0, 2)
        #     return output, attention_weights
        # else:
        #     output = output.permute(1, 0, 2)
        #     return output



class PositionalwiseFeedForward(nn.Cell):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ----------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.ffn = nn.SequentialCell([
            nn.Dense(input_size, d_ffn, weight_init='uniform', bias_init='uniform'),
            activation(),
            nn.Dropout(1.0-dropout),
            nn.Dense(d_ffn, input_size,weight_init='uniform', bias_init='uniform')
        ])
        self.transpose = ops.Transpose()

    def construct(self, x):
        # give a tensor of shap (time, batch, fea)
        x = self.transpose(x, (1, 0, 2))
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = self.transpose(x, (1, 0, 2))

        return x


if __name__ == '__main__':
    from mindspore import context
    target = "Ascend"
    device_id = 7

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    context.set_context(device_id=device_id)

    # inputs = mindspore.rand([8, 60, 512])
    shape = (8, 60, 512)
    uniformreal = ops.UniformReal(seed=2)
    inputs = uniformreal(shape)
    net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    outputs, attn = net(inputs, inputs, inputs)
    print(outputs.shape)
    # done
