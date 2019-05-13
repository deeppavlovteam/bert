from typing import Optional, Mapping, Tuple, Union

import tensorflow as tf

from .activations import gelu


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.layers = []
        for i in range(num_hidden_layers):
            self.layers.append(TransformerBlock(hidden_size=hidden_size,
                                                intermediate_size=intermediate_size,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate,
                                                intermediate_act_fn=intermediate_act_fn,
                                                layer_norm_epsilon=layer_norm_epsilon,
                                                name=f'layer_{i}'))

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:
        x = inputs
        for l in self.layers:
            x = l(x, mask=mask)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, name='attention')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon,
                                                             name='attention/output/LayerNorm')

        # point-wise feed forward network
        self.pff1 = tf.keras.layers.Dense(intermediate_size,
                                          activation=intermediate_act_fn,
                                          name='intermediate/dense')
        self.pff2 = tf.keras.layers.Dense(hidden_size, activation='relu', name='output/dense')

        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='output/LayerNorm')

    def call(self, inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:
        x = inputs
        attn_output, _ = self.mha({'query': x, 'key': x, 'value': x}, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.pff2(self.pff2(out1))  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.depth = hidden_size // num_heads

        # TODO: generalize naming to non-self attention
        self.wq = tf.keras.layers.Dense(hidden_size, name='self/query')
        self.wk = tf.keras.layers.Dense(hidden_size, name='self/key')
        self.wv = tf.keras.layers.Dense(hidden_size, name='self/value')

        self.dense = tf.keras.layers.Dense(hidden_size, name='output/dense')

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,
             inputs: Mapping[str, tf.Tensor],
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.shape(inputs['query'])[0]

        q = self.wq(inputs['query'])  # (batch_size, seq_len, d_model)
        k = self.wk(inputs['key'])  # (batch_size, seq_len, d_model)
        v = self.wv(inputs['value'])  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_v, hidden_size)
        output = self.dense(concat_attention)  # (batch_size, seq_len_v, hidden_size)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has shape depending on its type (padding / look-ahead) but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)

    return output, attention_weights
