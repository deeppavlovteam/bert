from typing import Sequence, Optional, Tuple

import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 attention_probs_dropout_prob: float = 0.1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.depth = hidden_size // num_heads

        self.wq = tf.keras.layers.Dense(hidden_size, name='self/query')
        self.wk = tf.keras.layers.Dense(hidden_size, name='self/key')
        self.wv = tf.keras.layers.Dense(hidden_size, name='self/value')
        self.dropout = tf.keras.layers.Dropout(rate=attention_probs_dropout_prob)
        self.dense = tf.keras.layers.Dense(hidden_size, name='output/dense')

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,
             x: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_weights = self.dropout(attention_weights, training=training)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.hidden_size))  # (batch_size, seq_len_v, hidden_size)
        output = self.dense(concat_attention)  # (batch_size, seq_len_v, hidden_size)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights. q, k, v must have matching leading dimensions.
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
