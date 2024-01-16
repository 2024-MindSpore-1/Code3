import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Attention(nn.Cell):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']
        self.hidden_weight = nn.Dense(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, pad_mode='pad')
        self.attention_weight = nn.Dense(512, self.attention_dim, has_bias=False)
        self.alpha_convert = nn.Dense(self.attention_dim, 1)

    def construct(self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        transpose = ops.Transpose()
        alpha_sum_trans = transpose(alpha_sum_trans,(0,2,3,1))
        coverage_alpha = self.attention_weight(alpha_sum_trans)
        cnn_features_trans = transpose(cnn_features_trans,(0,2,3,1))
        query_numpy = query.asnumpy()
        alpha_score = ops.tanh(ms.Tensor(query_numpy[:, None, None, :],ms.float32) + coverage_alpha + cnn_features_trans)
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = ops.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        energy_exp_numpy = energy_exp.asnumpy()
        alpha = energy_exp / ms.Tensor(energy_exp_numpy.sum(-1).sum(-1)[:,None,None] + 1e-10,ms.float32)
        alpha_numpy = alpha.asnumpy()
        alpha_sum = ms.Tensor(alpha_numpy[:,None,:,:],ms.float32) + alpha_sum
        context_vector = (ms.Tensor(alpha_numpy[:,None,:,:],ms.float32) * cnn_features).sum(-1).sum(-1)
        return context_vector, alpha, alpha_sum
