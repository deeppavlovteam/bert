from typing import Optional, Union

import tensorflow as tf

from .embeddings import BERTCombinedEmbedding
from .attention import MultiHeadAttention, MultiHeadSelfAttention
from .activations import gelu

# from .normalization import LayerNormalization
try:
    from tensorflow.keras.layers import LayerNormalization
except ImportError:
    from .normalization import LayerNormalization

# def layer_norm(input_tensor, name=None):
#   """Run layer normalization on the last dimension of the tensor."""
#   return tf.contrib.layers.layer_norm(
#       inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


class BERT(tf.keras.Model):
    """BERT body"""
    def __init__(self,
                 return_stack: Optional[bool] = None,
                 vocab_size: int = 119547,
                 token_type_vocab_size: int = 2,
                 sep_token_index: int = 103,
                 pad_token_index: int = 0,
                 emb_dropout_rate: float = 0.1,  # equal to hidden_dropout_prob in the original implementation
                 max_len: int = 512,
                 use_one_hot_embedding: bool = False,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,
                 layer_norm_epsilon: float = 1e-12,
                 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315
                 # is tf.keras version identical to the original layer_norm, which is copied above?
                 hidden_size: int = 768,

                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,  # TODO: clarify usage of this parameter
                 intermediate_act_fn: Union[str, callable] = gelu,
                 pooler_fc_size: int = 768,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.return_stack = return_stack
        self.pad_token_index = pad_token_index
        with tf.name_scope('embeddings'):
            self.emb = BERTCombinedEmbedding(#input_shape=max_len,
                                             vocab_size=vocab_size,
                                             token_type_vocab_size=token_type_vocab_size,
                                             sep_token_index=sep_token_index,
                                             output_dim=hidden_size,
                                             use_one_hot_embedding=use_one_hot_embedding,
                                             max_len=max_len,
                                             initializer_range=initializer_range,
                                             trainable_pos_embedding=trainable_pos_embedding,
                                             trainable=trainable,
                                             name='embeddings')
            self.embedding_dropout = tf.keras.layers.Dropout(rate=emb_dropout_rate, name='embeddings/dropout')
            self.embedding_layer_norm = LayerNormalization(epsilon=layer_norm_epsilon,
                                                           name='embeddings/LayerNorm')

        with tf.name_scope('encoder'):
            self.encoder = tf.keras.Sequential(name='encoder')
            for i in range(num_hidden_layers):
                with tf.name_scope(f'layer_{i}'):
                    self.encoder.add(TransformerBlock(hidden_size=hidden_size,
                                                      intermediate_size=intermediate_size,
                                                      num_heads=num_heads,
                                                      hidden_dropout_prob=hidden_dropout_prob,
                                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                      intermediate_act_fn=intermediate_act_fn,
                                                      layer_norm_epsilon=layer_norm_epsilon,
                                                      trainable=trainable,
                                                      name=f'layer_{i}'))

        self.pooler = tf.keras.layers.Dense(pooler_fc_size, activation='tanh', name='pooler/dense')

    # @staticmethod
    # def create_self_attention_mask_from_input_mask(input_mask):
    #     """
    #     Create 4D self attention mask (inverted, in order to simplify logits addition) from a 2D input mask.
    #
    #     Args:
    #         input_mask: 2D int tf.Tensor of shape [batch_size, seq_length].
    #
    #     Returns:
    #         float tf.Tensor of shape [batch_size, 1, seq_length, 1].
    #     """
    #     # inverted_mask = tf.cast(, tf.float32)
    #     return tf.cast(input_mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        embed = self.emb(inputs, training=training)
        emb_norm_do = self.embedding_dropout(self.embedding_layer_norm(embed), training=training)

        # attention_mask = self.create_self_attention_mask_from_input_mask(mask)
        if mask is None:
            mask = tf.cast(tf.not_equal(inputs, self.pad_token_index), tf.int32)
        enc = self.encoder(inputs=emb_norm_do, training=training, mask=mask)
        po = self.pooler(tf.squeeze(enc[:, 0:1, :], axis=1))

        if self.return_stack is None:
            return po
        elif self.return_stack:
            raise NotImplementedError('Currently all encoder layers output could not be obtained')
        else:
            return enc


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_heads: int = 12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.0,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.mhsa = MultiHeadAttention(hidden_size=hidden_size,
                                       num_heads=num_heads,
                                       attention_probs_dropout_prob=attention_probs_dropout_prob,
                                       name='attention')

        self.dense = tf.keras.layers.Dense(units=hidden_size, name='attention/output/dense')

        self.dropout1 = tf.keras.layers.Dropout(rate=hidden_dropout_prob,
                                                name='attention/output/dropout')
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon,
                                             name='attention/output/LayerNorm')

        # point-wise feed forward network
        self.pff1 = tf.keras.layers.Dense(units=intermediate_size,
                                          activation=intermediate_act_fn,
                                          name='intermediate/dense')
        self.pff2 = tf.keras.layers.Dense(units=hidden_size, name='output/dense')

        self.dropout2 = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon, name='output/LayerNorm')

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        attn_output = self.mhsa(inputs, inputs, inputs, mask=mask)
        attn_output = self.dropout1(self.dense(attn_output), training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.pff2(self.pff1(out1))
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        out2._keras_mask = mask  # workaround for mask propagation
        return out2

    def compute_mask(self,
                     inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape
