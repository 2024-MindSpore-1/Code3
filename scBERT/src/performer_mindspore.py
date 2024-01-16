import math
from contextlib import contextmanager
from functools import partial

import mindspore as ms
import numpy as np
import scipy.linalg as sl
from mindspore import Tensor, nn, ops

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    b, h, *_ = data.shape
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    ratio = (projection_matrix.shape[0] ** -0.5)
    projection = ops.stack([projection_matrix] * h, 0)
    projection = ops.stack([projection] * b, 0)
    projection = projection.astype(data.dtype)
    data_dash = ops.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    diag_data = data ** 2
    diag_data = ops.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (ops.exp(data_dash - diag_data - ops.max(data_dash, axis=-1, keepdims=True)[0]) + eps)
    else:
        data_dash = ratio * (ops.exp(data_dash - diag_data - ops.max(data_dash)[0]) + eps)

    return data_dash.astype(data.dtype)

def linear_attention(q, k, v):
    k_cumsum = ops.sum(k, dim=-2)
    D_inv = 1. / ops.einsum('...nd,...d->...n', q, k_cumsum.astype(q.dtype))
    context = ops.einsum('...nd,...ne->...de', k, v)
    out = ops.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

def orthogonal_matrix_chunk(cols):
    unstructured_block = np.random.randn(cols, cols)
    q, r = sl.qr(unstructured_block)
    return q.T

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns)
        block_list.append(q[:remaining_rows])
    final_matrix = np.concatenate(block_list)

    if scaling == 0:
        multiplier = np.random.randn(nb_rows, nb_columns)
        multiplier = np.linalg.norm(multiplier, axis=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * (nb_rows,)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    out = np.diag(multiplier) @ final_matrix
    out = Tensor(out)
    return out


class Always(nn.Cell):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def construct(self, *args, **kwargs):
        return self.val


class FastAttention(nn.Cell):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=self.dim_heads, scaling=self.ortho_scaling)
        self.projection_matrix = self.create_projection()

    def construct(self, q, k, v):
        create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix)
        q = create_kernel(q, is_query = True)
        k = create_kernel(k, is_query = False)
        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        return out


class SelfAttention(nn.Cell):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 nb_features=None,
                 qkv_bias=False,
                 dropout=0.,):
        super().__init__()
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.heads = heads
        self.fast_attention = FastAttention(dim_head, nb_features)

        self.to_q = nn.Dense(dim, inner_dim, has_bias=qkv_bias)
        self.to_k = nn.Dense(dim, inner_dim, has_bias=qkv_bias)
        self.to_v = nn.Dense(dim, inner_dim, has_bias=qkv_bias)
        self.to_out = nn.Dense(inner_dim, dim)
        self.dropout = nn.Dropout(keep_prob=1.0-dropout)

    def rearrange(self, x, h):
        ss = x.shape[:-1] + (h, int(x.shape[-1]/h))
        x = ops.reshape(x, ss)
        x = ops.swapaxes(x, -3, -2)
        return x

    def construct(self, x, **kwargs):
        h = self.heads
        context = x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = self.rearrange(q, h)
        k = self.rearrange(k, h)
        v = self.rearrange(v, h)
        out = self.fast_attention(q, k, v)
        out = ops.swapaxes(out, -3, -2)
        out = ops.reshape(out, (out.shape[0], out.shape[1], -1))
        out = self.to_out(out)
        out = self.dropout(out)
        return out


class Chunk(nn.Cell):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def construct(self, x, **kwargs):
        if self.chunks == 1:
            x = self.fn(x, **kwargs)
            return x
        chunks = ops.chunk(x, self.chunks, axis=self.dim)
        x = ops.cat([self.fn(c, **kwargs) for c in chunks], axis=self.dim)
        return x


class FeedForward(nn.Cell):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        if activation is None:
            activation = nn.GELU()

        self.glu = glu
        self.w1 = nn.Dense(dim, dim * mult * (2 if glu else 1))
        self.act = activation
        self.dropout = nn.Dropout(keep_prob=1.0-dropout)
        self.w2 = nn.Dense(dim * mult, dim)

    def construct(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x = self.w1(x)
            x, v = ops.chunk(x, 2, dim=-1)
            x = self.act(x) * v
        x = self.dropout(x)
        x = self.w2(x)
        return x


class PreLayerNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        ddim = cast_tuple(dim)
        self.norm = nn.LayerNorm(ddim)
        self.fn = fn
    def construct(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x


class Gene2VecPositionalEmbedding(nn.Cell):
    def __init__(self):
        super().__init__()
        gene2vec_weight = np.load('./data/gene2vec_16906.npy')
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        self.gene2vec_weight = Tensor(gene2vec_weight)

    def construct(self, x):
        t = ops.arange(x.shape[1])
        t = ops.gather(self.gene2vec_weight, t, axis=0)
        return t


class Performer(nn.Cell):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 local_attn_heads = 0,
                 ff_mult = 4,
                 nb_features = None,
                 ff_chunks = 1,
                 ff_glu = False,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 qkv_bias = True,
                 ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        wrapper_fn = partial(PreLayerNorm, dim)
        self.layers = nn.CellList([])
        for _, local_heads in zip(range(depth), local_attn_heads):
            _layers = nn.CellList([])
            _layers.append(wrapper_fn(SelfAttention(dim, heads=heads, dim_head=dim_head, nb_features=nb_features,
                                                    dropout=attn_dropout, qkv_bias=qkv_bias)))
            _layers.append(wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)))
            self.layers.append(_layers)

    def construct(self, x, pos_emb):
        for f, g in self.layers:
            x = x + f(x, pos_emb=pos_emb)
            x = x + g(x)
        return x


class PerformerLM(nn.Cell):
    def __init__(self,
                 num_tokens,
                 max_seq_len,
                 dim,
                 depth,
                 heads,
                 dim_head = 64,
                 local_attn_heads = 0,
                 ff_mult = 4,
                 nb_features = None,
                 ff_chunks = 1,
                 ff_glu = False,
                 emb_dropout = 0.,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 tie_embed = False,
                 qkv_bias = False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = Gene2VecPositionalEmbedding()
        self.layer_pos_emb = Always(None)
        self.dropout = nn.Dropout(keep_prob=1.0-emb_dropout)
        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, ff_mult, nb_features,
                                   ff_chunks, ff_glu, ff_dropout, attn_dropout, qkv_bias)
        ddim = cast_tuple(dim)
        self.norm = nn.LayerNorm(ddim)
        self.to_out = nn.Dense(dim, num_tokens) if not tie_embed else None

    def construct(self, x):
        x = self.token_emb(x)
        y = self.pos_emb(x)
        x = x + y
        x = x.astype(ms.float32)
        x = self.dropout(x)
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb=layer_pos_emb)
        x = self.norm(x)
        x = self.to_out(x)
        return x