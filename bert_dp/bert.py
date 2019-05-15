from typing import Optional, Union

import tensorflow as tf

from .embeddings import BERTCombinedEmbedding
from .attention import MultiHeadSelfAttention
from .activations import gelu
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
                 vocab_size: int = 119547,
                 sep_token_index: int = 103,
                 emb_dropout_rate: float = 0.1,
                 max_len: int = 512,
                 use_one_hot_embedding: bool = False,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
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
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.emb = BERTCombinedEmbedding(vocab_size=vocab_size,
                                         sep_token_index=sep_token_index,
                                         output_dim=hidden_size,
                                         use_one_hot_embedding=use_one_hot_embedding,
                                         max_len=max_len,
                                         initializer_range=initializer_range,
                                         trainable_pos_embedding=trainable_pos_embedding,
                                         name='embeddings')
        self.embedding_dropout = tf.keras.layers.Dropout(rate=emb_dropout_rate, name='embeddings/dropout')
        self.embedding_layer_norm = LayerNormalization(epsilon=layer_norm_epsilon,
                                                       name='embeddings/LayerNorm')

        self.encoder = tf.keras.Sequential(name='encoder')
        for i in range(num_hidden_layers):
            self.encoder.add(TransformerBlock(hidden_size=hidden_size,
                                              intermediate_size=intermediate_size,
                                              num_heads=num_heads,
                                              hidden_dropout_prob=hidden_dropout_prob,
                                              attention_probs_dropout_prob=attention_probs_dropout_prob,
                                              intermediate_act_fn=intermediate_act_fn,
                                              layer_norm_epsilon=layer_norm_epsilon,
                                              name=f'layer_{i}'))

        # what about tf.keras.layers.TimeDistributed?
        self.pooler = tf.keras.layers.Dense(pooler_fc_size, name='pooler/dense')

    @staticmethod
    def create_self_attention_mask_from_input_mask(input_mask):
        """
        Create 4D self attention mask (inverted, in order to simplify logits addition) from a 2D input mask.

        Args:
            input_mask: 2D int tf.Tensor of shape [batch_size, seq_length].

        Returns:
            float tf.Tensor of shape [batch_size, 1, 1, seq_length].
        """
        inverted_mask = tf.cast(tf.math.equal(input_mask, 0), tf.float32)
        return inverted_mask[:, tf.newaxis, tf.newaxis, :]

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        embed = self.emb(inputs, mask=mask)
        emb_norm_do = self.embedding_dropout(self.embedding_layer_norm(embed), training=training)
        attention_mask = self.create_self_attention_mask_from_input_mask(mask)
        enc = self.encoder(inputs=emb_norm_do, mask=attention_mask)
        return self.pooler(enc)


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

        self.mhsa = MultiHeadSelfAttention(hidden_size=hidden_size,
                                           num_heads=num_heads,
                                           attention_probs_dropout_prob=attention_probs_dropout_prob,
                                           name='attention')

        self.dropout1 = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon,
                                             name='attention/output/LayerNorm')

        # point-wise feed forward network
        self.pff1 = tf.keras.layers.Dense(units=intermediate_size,
                                          activation=intermediate_act_fn,
                                          name='intermediate/dense')
        self.pff2 = tf.keras.layers.Dense(units=hidden_size, activation='relu', name='output/dense')

        self.dropout2 = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon, name='output/LayerNorm')

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        attn_output, _ = self.mhsa(inputs, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.pff2(self.pff1(out1))  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
