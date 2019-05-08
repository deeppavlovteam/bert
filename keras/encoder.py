from typing import Optional

import tensorflow as tf

from .activations import gelu


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12) -> None:
        super().__init__(name='encoder')
        self.layers = []
        for i in range(num_hidden_layers):
            self.layers.append(TransformerUnit(hidden_size=hidden_size,
                                               intermediate_size=intermediate_size,
                                               name=f'layer_{i}'))

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        for l in self.layers:
            x = l(x, mask=mask)
        return x


class TransformerUnit(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_heads: int = 12,
                 rate=0.1,
                 **kwargs) -> None:
        kwargs['name'] = kwargs.get('name', 'layer')
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # point-wise feed forward network
        self.pff1 = tf.keras.layers.Dense(intermediate_size, activation=gelu)
        self.pff2 = tf.keras.layers.Dense(hidden_size, activation='relu')

        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs):
        x = inputs
        attn_output, _ = self.mha((x, x, x), mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.pff2(self.pff2(out1))  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, **kwargs) -> None:
        kwargs['name'] = kwargs.get('name', 'attention/self')
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert hidden_size % num_heads == 0

        self.depth = hidden_size // num_heads

        self.wq = tf.keras.layers.Dense(hidden_size, name='query')
        self.wk = tf.keras.layers.Dense(hidden_size, name='key')
        self.wv = tf.keras.layers.Dense(hidden_size, name='value')

        self.dense = tf.keras.layers.Dense(hidden_size, name='output')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None, mask=None):
        v, k, q = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_v, hidden_size)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, hidden_size)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

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

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)

    return output, attention_weights
