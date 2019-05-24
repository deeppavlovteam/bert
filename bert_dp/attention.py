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
        self.size_per_head = hidden_size // num_heads

        self.wq = tf.keras.layers.Dense(hidden_size, name='self/query')
        self.wk = tf.keras.layers.Dense(hidden_size, name='self/key')
        self.wv = tf.keras.layers.Dense(hidden_size, name='self/value')
        self.dropout = tf.keras.layers.Dropout(rate=attention_probs_dropout_prob)

    def split_heads(self, x, batch_size):
        """
        Equivalent of transpose_for_scores in the original implementation.
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.size_per_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def create_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """Create 3D attention mask from a 2D tensor mask.

        Args:
          from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
          to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
          float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        from_shape = tf.shape(from_tensor)
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        to_shape = tf.shape(to_mask)
        to_seq_length = to_shape[1]

        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

        # We don't assume that `from_tensor` is a mask (although it could be). We
        # don't actually care if we attend *from* padding tokens (only *to* padding)
        # tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask

    def call(self,
             x: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        q = self.wq(x)  # (B, F, N*H)
        k = self.wk(x)  # (B, F, N*H)
        v = self.wv(x)  # (B, F, N*H)

        q = self.split_heads(q, batch_size)  # (B, N, F, H)
        k = self.split_heads(k, batch_size)  # (B, N, T, H)
        v = self.split_heads(v, batch_size)  # (B, N, T, H)

        attention_scores = tf.matmul(q, k, transpose_b=True)

        attention_scores = tf.multiply(attention_scores,
                                       1.0 / tf.math.sqrt(float(self.size_per_head)))

        if mask is not None:
            attention_mask = self.create_attention_mask_from_input_mask(x, mask)
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # `context_layer` = [B, N, F, H]
        output = tf.matmul(attention_probs, v)

        # `context_layer` = [B, F, N, H]
        output = tf.transpose(output, [0, 2, 1, 3])

        output = tf.reshape(output, [batch_size, seq_len, self.hidden_size])

        return output

#
# def scaled_dot_product_attention(q, k, v, mask):
#     """
#     Calculate the attention weights. q, k, v must have matching leading dimensions.
#     The mask has shape depending on its type (padding / look-ahead) but it must be broadcastable for addition.
#
#     Args:
#         q: query shape == (..., seq_len_q, depth)
#         k: key shape == (..., seq_len_k, depth)
#         v: value shape == (..., seq_len_v, depth)
#         mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
#
#     Returns:
#         output, attention_weights
#     """
#
#     attention_scores = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
#
#     # scale attention_scores
#     dk = tf.cast(tf.shape(k)[-1], tf.float32)
#     scaled_attention_logits = attention_scores / tf.math.sqrt(dk)
#
#     # add the mask to the scaled tensor.
#     if mask is not None:
#         scaled_attention_logits += (mask * -1e9)
#
#     # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
#     attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
#
#     output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)
#
#     return output, attention_weights
