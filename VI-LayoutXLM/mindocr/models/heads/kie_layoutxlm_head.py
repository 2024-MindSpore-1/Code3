import math
from dataclasses import dataclass

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal, Zero

__all__ = ["LayoutXLMForSer"]

def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0.).astype(ms.int32) * num_buckets
        n = ops.abs(relative_position)
    else:
        n = ops.maximum(-relative_position, ops.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = n < max_exact

    val_if_large = max_exact + (ops.log(n.astype(ms.float32) / max_exact) / \
                                math.log(max_distance / max_exact) * \
                                (num_buckets - max_exact)).astype(ms.int32)

    val_if_large = ops.minimum(val_if_large, ops.full_like(val_if_large, num_buckets - 1))

    ret += ops.where(is_small, n, val_if_large)
    return ret

@dataclass
class ViLayoutXLMPretrainedConfig:
    attention_probs_dropout_prob = 0.1
    bos_token_id = 0
    coordinate_size = 128
    eos_token_id = 2
    fast_qkv = False
    gradient_checkpointing = False
    has_relative_attention_bias = False
    has_spatial_attention_bias = False
    has_visual_segment_embedding = True
    use_visual_backbone = False
    hidden_act = "gelu"
    hidden_dropout_prob = 0.1
    hidden_size = 768
    image_feature_pool_shape = (7, 7, 256)
    initializer_range = 0.02
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    max_2d_position_embeddings = 1024
    max_position_embeddings = 514
    max_rel_2d_pos = 256
    max_rel_pos = 128
    model_type = "layoutlxlm"
    num_attention_heads = 12
    num_hidden_layers = 12
    output_past = True
    pad_token_id = 1
    shape_size = 128
    rel_2d_pos_bias = 64
    rel_pos_bias = 32
    type_vocab_size = 1
    vocab_size = 250002


class LayoutXLMEmbeddings(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layernorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.position_ids = Parameter(ops.arange(0, config.max_position_embeddings, dtype=ms.int32).broadcast_to((1, -1)),
                                      name="position_ids", requires_grad=False)
        
    def _calc_spatial_position_embeddings(self, bbox):
        bbox_0 = bbox[:, :, 0]
        bbox_1 = bbox[:, :, 1]
        bbox_2 = bbox[:, :, 2]
        bbox_3 = bbox[:, :, 3]
        left_position_embeddings = self.x_position_embeddings(bbox_0)
        upper_position_embeddings = self.y_position_embeddings(bbox_1)
        right_position_embeddings = self.x_position_embeddings(bbox_2)
        lower_position_embeddings = self.y_position_embeddings(bbox_3)

        h_position_embeddings = self.h_position_embeddings(bbox_3 - bbox_1)
        w_position_embeddings = self.w_position_embeddings(bbox_2 - bbox_0)

        spatial_position_embeddings = ops.concat(
            (
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ),
            axis=-1,
        )
        return spatial_position_embeddings
    
    def construct(self, input_ids, bbox=None, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = ops.ones_like(input_ids, dtype=ms.int32)
            seq_length = ops.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids = ops.stop_gradient(position_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids, dtype=ms.int32)

        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        bbox_0 = bbox[:, :, 0]
        bbox_1 = bbox[:, :, 1]
        bbox_2 = bbox[:, :, 2]
        bbox_3 = bbox[:, :, 3]
        left_position_embeddings = self.x_position_embeddings(bbox_0)
        upper_position_embeddings = self.y_position_embeddings(bbox_1)
        right_position_embeddings = self.x_position_embeddings(bbox_2)
        lower_position_embeddings = self.y_position_embeddings(bbox_3)

        h_position_embeddings = self.h_position_embeddings(bbox_3 - bbox_1)
        w_position_embeddings = self.w_position_embeddings(bbox_2 - bbox_0)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            input_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )

        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class LayoutXLMSelfAttention(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of \
                  the number of attention heads {config.num_attention_heads}"
            )
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scale = Tensor(self.attention_head_size ** -0.5)
        
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if config.fast_qkv:
            self.qkv_linear = nn.Dense(config.hidden_size, 3 * self.all_head_size, has_bias=False)
            self.q_bias = Parameter(initializer(Zero(), shape=(1, 1, self.all_head_size), dtype=ms.float32))
            self.v_bias = Parameter(initializer(Zero(), shape=(1, 1, self.all_head_size), dtype=ms.float32))
        else:
            self.query = nn.Dense(config.hidden_size, self.all_head_size)
            self.key = nn.Dense(config.hidden_size, self.all_head_size)
            self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [self.num_attention_heads, self.attention_head_size]

        x = ops.reshape(x, new_x_shape)
        return ops.transpose(x, (0, 2, 1, 3))

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = ops.chunk(qkv, 3, axis=-1)
            if q.ndim == self.q_bias.ndim:
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1, ) * (q.dim - 1) + (-1,)
                q = q + ops.reshape(self.q_bias, _sz)
                v = v + ops.reshape(self.v_bias, _sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None
    ):
        q, k, v = self.compute_qkv(hidden_states)

        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = ops.mul(query_layer, self.scale)
        attention_scores = ops.matmul(query_layer, ops.transpose(key_layer, (0, 1, 3, 2)))

        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos

        bool_attention_mask = attention_mask.bool()
        bool_attention_mask = ops.stop_gradient(bool_attention_mask)
        attention_scores_shape = ops.shape(attention_scores)
        attention_scores = ops.where(
            bool_attention_mask.broadcast_to(attention_scores_shape),
            ops.ones(attention_scores_shape) * float("-1e10"),
            attention_scores
        )
        attention_probs = ops.softmax(attention_scores, axis=-1)

        attention_probs = self.dropout(attention_probs)
        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = ops.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = list(context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = ops.reshape(context_layer, new_context_layer_shape)

        if output_attentions:
            outputs = [context_layer, attention_probs]
        else:
            outputs = [context_layer]
        return outputs
    

class LayoutXLMAttnOutput(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMAttnOutput, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        output = self.dense(hidden_states)
        output = self.dropout(output)
        output = self.layernorm(output + input_tensor)
        return output


class LayoutXLMAttention(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMAttention, self).__init__()
        self.attn = LayoutXLMSelfAttention(config)
        self.output = LayoutXLMAttnOutput(config)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None
    ):
        attn_outputs = self.attn(
            hidden_states,
            attention_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos
        )
        output = self.output(attn_outputs[0], hidden_states)
        if output_attentions:
            outputs = [output] + attn_outputs[1:]
        else:
            outputs = [output]
        return outputs
    

class LayoutXLMIntermediate(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMIntermediate, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        else:
            raise ValueError("hidden_act should be `gelu`.")

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class LayoutXLMOutput(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMOutput, self).__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.layernorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        output = self.dense(hidden_states)
        output = self.dropout(output)
        output = self.layernorm(output + input_tensor)
        return output


class LayoutXLMLayer(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMLayer, self).__init__()
        self.seq_len_dim = 1
        self.attention = LayoutXLMAttention(config)
        self.intermediate = LayoutXLMIntermediate(config)
        self.output = LayoutXLMOutput(config)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions,
            rel_pos,
            rel_2d_pos
        )
        attention_output = self_attention_outputs[0]
        layer_output = self.feed_forward_chunk(attention_output)

        if output_attentions:
            outputs = [layer_output] + self_attention_outputs[1:]
        else:
            outputs = [layer_output]
        return outputs


class LayoutXLMEncoder(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(LayoutXLMEncoder, self).__init__()
        self.config = config
        self.layer = nn.CellList([LayoutXLMLayer(config) for _ in range(config.num_hidden_layers)])

        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        if self.has_relative_attention_bias:
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = config.rel_pos_bias
            self.rel_pos_onehot_size = config.rel_pos_bias
            self.rel_pos_bias = nn.Dense(self.rel_pos_onehot_size, config.num_attention_heads, has_bias=False)
            self.onehot_1d = nn.OneHot(depth=self.rel_pos_onehot_size)

        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bias = config.rel_2d_pos_bias
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bias
            self.rel_pos_x_bias = nn.Dense(self.rel_2d_pos_onehot_size, config.num_attention_heads, has_bias=False)
            self.rel_pos_y_bias = nn.Dense(self.rel_2d_pos_onehot_size, config.num_attention_heads, has_bias=False)
            self.onehot_2d = nn.OneHot(depth=self.rel_2d_pos_onehot_size)


    def _cal_1d_pos_emb(self, hidden_states, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bias,
            max_distance=self.max_rel_pos
        )
        rel_pos = self.onehot_1d(rel_pos).astype(hidden_states.dtype)
        rel_pos = ops.transpose(self.rel_pos_bias(rel_pos), (0, 3, 1, 2))
        return rel_pos

    def _cal_2d_pos_emb(self, hidden_states, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bias,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bias,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = self.onehot_2d(rel_pos_x).astype(hidden_states.dtype)
        rel_pos_y = self.onehot_2d(rel_pos_y).astype(hidden_states.dtype)
        rel_pos_x = ops.transpose(self.rel_pos_x_bias(rel_pos_x), (0, 3, 1, 2))
        rel_pos_y = ops.transpose(self.rel_pos_y_bias(rel_pos_y), (0, 3, 1, 2))
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        bbox=None,
        position_ids=None
    ):
        all_hidden_states = () if output_hidden_states else None

        rel_pos = self._cal_1d_pos_emb(hidden_states, position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(hidden_states, bbox) if self.has_spatial_attention_bias else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
                rel_pos=rel_pos,
                rel_2d_pos=rel_2d_pos,
            )

            hidden_states = layer_outputs[0]

        return hidden_states
    

class LayoutXLMPooler(nn.Cell):
    def __init__(
        self,
        hidden_size,
        with_pool
    ):
        super(LayoutXLMPooler, self).__init__()
        self.dense = nn.Dense(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.with_pool = with_pool

    def construct(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.with_pool:
            pooled_output = self.activation(pooled_output)
        return pooled_output
        

class VisualBackbone(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig
    ):
        super(VisualBackbone, self).__init__()
        raise NotImplementedError("The visual backbone for LayoutXLM is not implemented.")


class LayoutXLMModel(nn.Cell):
    def __init__(
        self,
        config: ViLayoutXLMPretrainedConfig,
        with_pool=True,
        **kwargs
    ):
        super(LayoutXLMModel, self).__init__()
        self.config = config
        self.use_visual_backbone = config.use_visual_backbone
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutXLMEmbeddings(config)

        if self.use_visual_backbone:
            self.visual = VisualBackbone(config)
            self.visual_proj = nn.Dense(config.image_feature_pool_shape[-1], config.hidden_size)

        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = Parameter(nn.Embedding(1, config.hidden_size,
                                                                   embedding_table="xavier_uniform").embedding_table[0])
        self.visual_layernorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.encoder = LayoutXLMEncoder(config)
        self.pooler = LayoutXLMPooler(config.hidden_size, with_pool)
        self.image_feature_pool_shape = config.image_feature_pool_shape
        self.image_feature_pool_shape_size = config.image_feature_pool_shape[0] * config.image_feature_pool_shape[1]

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.layernorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings
    
    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, visual_shape):
        visual_bbox_x = ops.arange(0, 1000 * (image_feature_pool_shape[1] + 1),
                                   1000, dtype=bbox.dtype) // image_feature_pool_shape[1]
        visual_bbox_y = ops.arange(0, 1000 * (image_feature_pool_shape[0] + 1),
                                   1000, dtype=bbox.dtype) // image_feature_pool_shape[0]

        expand_shape = image_feature_pool_shape[0:2]
        visual_bbox = ops.stack([
            visual_bbox_x[:-1].broadcast_to(expand_shape),
            visual_bbox_y[:-1].broadcast_to(expand_shape[::-1]).transpose((1, 0)),
            visual_bbox_x[1:].broadcast_to(expand_shape),
            visual_bbox_y[1:].broadcast_to(expand_shape[::-1]).transpose((1, 0)),
        ] ,axis=-1).reshape((expand_shape[0] * expand_shape[1], ops.shape(bbox)[-1]))
        visual_bbox = visual_bbox.broadcast_to((visual_shape[0], visual_bbox.shape[0], visual_bbox.shape[1]))
        return visual_bbox
    
    def _calc_img_embeddings(self, image, bbox, position_ids):
        use_image_info = self.use_visual_backbone and image is not None
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        if use_image_info:
            visual_embeddings = self.visual_proj(self.visual(image.astype(ms.float32)))
            embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        else:
            embeddings = position_embeddings + spatial_position_embeddings

        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_layernorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings
    
    def construct(
        self,
        input_ids=None,
        bbox=None,
        image=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False
    ):
        input_shape = ops.shape(input_ids)
        visual_shape = list(input_shape)
        visual_shape[1] = self.image_feature_pool_shape_size
        visual_bbox = self._calc_visual_bbox(self.image_feature_pool_shape, bbox, visual_shape)

        final_bbox = ops.concat((bbox, visual_bbox), axis=1)
        if attention_mask is None:
            attention_mask = ops.ones(visual_shape, dtype=ms.int32)

        if self.use_visual_backbone:
            visual_attention_mask = ops.ones(visual_shape, dtype=ms.int32)
        else:
            visual_attention_mask = ops.zeros(visual_shape, dtype=ms.int32)

        attention_mask = attention_mask.astype(visual_attention_mask.dtype)
        final_attention_mask = ops.concat((attention_mask, visual_attention_mask), axis=1)

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=ms.int32)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.broadcast_to(input_shape)

        visual_position_ids = ops.arange(0, visual_shape[1], dtype=ms.int32).broadcast_to((input_shape[0], visual_shape[1]))
        final_position_ids = ops.concat((position_ids, visual_position_ids), axis=1)

        if bbox is None:
            bbox = ops.zeros(input_shape + (4,))

        text_layout_emb = self._calc_text_embeddings(input_ids, bbox, position_ids, token_type_ids)
        visual_emb = self._calc_img_embeddings(image, visual_bbox, visual_position_ids)
        final_emb = ops.concat((text_layout_emb, visual_emb), axis=1)
        
        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_output = self.encoder(
            final_emb,
            extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            bbox=final_bbox,
            position_ids=final_position_ids
        )
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


class LayoutXLMForTokenClassification(nn.Cell):
    def __init__(
        self,
        layoutxlm,
        num_classes=2,
        dropout=None
    ):
        super(LayoutXLMForTokenClassification, self).__init__()
        self.num_classes = num_classes
        if isinstance(layoutxlm, ViLayoutXLMPretrainedConfig):
            self.layoutxlm = LayoutXLMModel(layoutxlm)
        else:
            self.layoutxlm = layoutxlm

        self.dropout = nn.Dropout(p=dropout if dropout is not None else self.layoutxlm.config.hidden_dropout_prob)
        self.classifier = nn.Dense(self.layoutxlm.config.hidden_size, num_classes)
        self._init_weights(self.classifier)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Dense):
            layer.weight.set_data(
                initializer(Normal(sigma=0.02), layer.weight.shape, layer.weight.dtype)
            )

    def construct(
        self,
        image=None,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
    ):
        outputs = self.layoutxlm(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask
        )
        seq_length = input_ids.shape[1]
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


class NLPBaseModel(nn.Cell):
    def __init__(
        self, 
        base_model_class, 
        model_class, 
        mode="vi", 
        type="ser",
        pretrained=True, 
        checkpoints=None,
        **kwargs
    ):
        super(NLPBaseModel, self).__init__()
        if checkpoints is not None:
            raise NotImplementedError("Loading checkpoint for LayoutXLM is not implemented.")
        else:
            config = ViLayoutXLMPretrainedConfig()
            base_model = base_model_class(config)
            if pretrained:
                raise NotImplementedError("The pretrained model for LayoutXLM is not implemented.")
            if type == "ser":
                self.model = model_class(base_model, num_classes=kwargs["num_classes"], dropout=None)
            else:
                self.model = model_class(base_model, dropout=None)
        
        self.out_channels = 1
        self.use_visual_backbone = True


class LayoutXLMForSer(NLPBaseModel):
    def __init__(
        self,
        num_classes,
        pretrained=False,
        checkpoints=None,
        mode="vi",
        **kwargs
    ):
        super(LayoutXLMForSer, self).__init__(
            LayoutXLMModel,
            LayoutXLMForTokenClassification,
            mode,
            "ser",
            pretrained,
            checkpoints,
            num_classes=num_classes
        )
        self.use_visual_backbone = self.model.layoutxlm.use_visual_backbone

    def construct(self, x, others):
        if self.use_visual_backbone:
            image = x
        else:
            image = None
        
        output = self.model(
            image=image,
            input_ids=others[0],
            bbox=others[1],
            attention_mask=others[2],
            token_type_ids=others[3],
            position_ids=None,
        )
        
        return output
