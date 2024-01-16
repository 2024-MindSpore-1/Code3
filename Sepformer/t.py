
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

        self.nhead = nhead
        self.d_model = d_model
        self.dropout = float(dropout)

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
if __name__ == '__main__':
    from mindspore import context
    target = "Ascend"
    device_id = 6

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
