import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from models.attention import Attention
import math
import numpy as np
from counting_utils import gen_counting_label


class PositionEmbeddingSine(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, x, mask):
        y_embed = mask.cumsum(1, dtype=ms.float32)
        x_embed = mask.cumsum(2, dtype=ms.float32)
        if self.normalize:
            y_embed_numpy = y_embed.asnumpy()
            x_embed_numpy = x_embed.asnumpy()
            eps = 1e-6
            y_embed_numpy = y_embed_numpy / (y_embed_numpy[:, -1:, :] + eps) * self.scale
            x_embed_numpy = x_embed_numpy / (x_embed_numpy[:, :, -1:] + eps) * self.scale
        
        dim_t = ms.numpy.arange(self.num_pos_feats, dtype=ms.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t_numpy = dim_t.asnumpy()
        
        pos_x = x_embed_numpy[:, :, :, None] / dim_t_numpy
        pos_y = y_embed_numpy[:, :, :, None] / dim_t_numpy
        stack = ops.Stack(4)
        
        pos_x = stack([ops.sin(ms.Tensor(pos_x[:, :, :, 0::2],ms.float32)), ops.cos(ms.Tensor(pos_x[:, :, :, 1::2],ms.float32))])
        pos_y = stack([ops.sin(ms.Tensor(pos_y[:, :, :, 0::2],ms.float32)), ops.cos(ms.Tensor(pos_y[:, :, :, 1::2],ms.float32))])
        a,b,c,d,e = pos_x.shape
        pos_x = pos_x.reshape((a,b,c,d*e))
        a,b,c,d,e = pos_y.shape
        pos_y = pos_y.reshape((a,b,c,d*e))
        concat_op = ops.Concat(axis=3)
        pos = concat_op((pos_y, pos_x))
        transpose = ops.Transpose()
        pos = transpose(pos,(0, 3, 1, 2))
        return pos


class AttDecoder(nn.Cell):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.word_num = params['word_num']
        self.counting_num = params['counting_decoder']['out_channel']

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        # init hidden state
        self.init_weight = nn.Dense(self.out_channel, self.hidden_size)
        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        # attention
        self.word_attention = Attention(params)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim,
                                              kernel_size=params['attention']['word_conv_kernel'],
                                              padding=params['attention']['word_conv_kernel']//2, has_bias=True, pad_mode='pad')

        self.word_state_weight = nn.Dense(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Dense(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Dense(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Dense(self.counting_num, self.hidden_size)
        self.word_convert = nn.Dense(self.hidden_size, self.word_num)

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def construct(self, cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=True):
        batch_size, num_steps = labels.shape
        height, width = cnn_features.shape[2:]
        zeros = ops.Zeros()
        word_probs = np.zeros((batch_size, num_steps, self.word_num))
        images_mask_numpy = images_mask.asnumpy()
        images_mask = ms.Tensor(images_mask_numpy[:, :, ::self.ratio, ::self.ratio],ms.float32)
        word_alpha_sum = zeros((batch_size, 1, height, width),ms.float32)
        word_alphas = np.zeros((batch_size, num_steps, height, width))
        hidden = self.init_hidden(cnn_features, images_mask)
        counting_context_weighted = self.counting_context_weight(counting_preds)
        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        images_mask_numpy = images_mask.asnumpy()
        pos = position_embedding(cnn_features_trans, ms.Tensor(images_mask_numpy[:,0,:,:],ms.float32))
        cnn_features_trans = cnn_features_trans + pos
        ones = ops.Ones()
        if is_train:
            for i in range(num_steps):
                labels_numpy = labels.asnumpy()
                word_embedding = self.embedding(ms.Tensor(labels_numpy[:, i-1],ms.int32)) if i else self.embedding(ones((batch_size),ms.int32))
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)
                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted
                word_prob = self.word_convert(word_out_state)
                word_prob_numpy = word_prob.asnumpy()
                word_alpha_numpy = word_alpha.asnumpy()
                word_probs[:, i] = word_prob_numpy
                word_alphas[:, i] = word_alpha_numpy
        else:
            word_embedding = self.embedding(ones((batch_size),ms.int32))
            for i in range(num_steps):
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)
                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted
                word_prob = self.word_convert(word_out_state)
                word,_ = ops.max(word_prob, 1)
                word_embedding = self.embedding(word.astype(ms.int32))
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
        return ms.Tensor(word_probs,ms.float32), ms.Tensor(word_alphas,ms.float32)

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return ops.tanh(average)
