from typing import Mapping, Optional, Union

import tensorflow as tf

from .embeddings import BERTCombinedEmbedding
from .encoder import TransformerEncoder
from .activations import gelu
from .normalization import LayerNormalization


# def layer_norm(input_tensor, name=None):
#   """Run layer normalization on the last dimension of the tensor."""
#   return tf.contrib.layers.layer_norm(
#       inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


class BERT(tf.keras.Model):
    """BERT body"""
    def __init__(self,
                 use_one_hot_embedding: bool = False,
                 emb_dropout_rate: float = 0.1,
                 vocab_size: int = 119547,
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
                 layer_norm_epsilon: float = 1e-12,
                 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315
                 # is tf.keras version identical to the original layer_norm, which is copied above?
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 pooler_fc_size: int = 768,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.emb = BERTCombinedEmbedding(output_dim=hidden_size,
                                         use_one_hot_embedding=use_one_hot_embedding,
                                         vocab_size=vocab_size,
                                         max_len=max_len,
                                         initializer_range=initializer_range,
                                         trainable_pos_embedding=trainable_pos_embedding,
                                         name='embeddings')
        self.embedding_dropout = tf.keras.layers.Dropout(rate=emb_dropout_rate, name='embeddings/dropout')
        self.embedding_layer_norm = LayerNormalization(epsilon=layer_norm_epsilon,
                                                       name='embeddings/LayerNorm')
        self.encoder = TransformerEncoder(hidden_size=hidden_size,
                                          intermediate_size=intermediate_size,
                                          num_hidden_layers=num_hidden_layers,
                                          num_heads=num_heads,
                                          dropout_rate=dropout_rate,
                                          intermediate_act_fn=intermediate_act_fn,
                                          layer_norm_epsilon=layer_norm_epsilon,
                                          name='encoder')
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
             inputs: Mapping[str, tf.Tensor],
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:
        embed = self.emb(inputs['segment_ids'], inputs['pos_ids'], inputs['token_ids'], mask=mask)
        emb_norm_do = self.embedding_dropout(self.embedding_layer_norm(embed))
        attention_mask = self.create_self_attention_mask_from_input_mask(mask)
        enc = self.encoder(inputs=emb_norm_do, mask=attention_mask)
        return self.pooler(enc)
