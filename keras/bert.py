from typing import Mapping, Optional, Union

import tensorflow as tf
from .embeddings import BERTCombinedEmbedding
from .encoder import TransformerEncoder
from .activations import gelu


class BERT(tf.keras.layers.Layer):
    """BERT body"""
    def __init__(self,
                 use_one_hot_embedding: bool = False,
                 emb_dropout_rate: float = 0.1,
                 vocab_size: int = 119547,
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
                 layer_norm_epsilon: float = 1e-12,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 pooler_fc_size: int = 768,
                 **kwargs) -> None:
        # hardcoded name!
        super().__init__(name='bert', **kwargs)
        self.emb = BERTCombinedEmbedding(output_dim=hidden_size,
                                         use_one_hot_embedding=use_one_hot_embedding,
                                         dropout_rate=emb_dropout_rate,
                                         vocab_size=vocab_size,
                                         max_len=max_len,
                                         initializer_range=initializer_range,
                                         trainable_pos_embedding=trainable_pos_embedding,
                                         layer_norm_epsilon=layer_norm_epsilon,
                                         name='embeddings')
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
        embed = self.emb(inputs)
        attention_mask = self.create_self_attention_mask_from_input_mask(mask)
        enc = self.encoder(inputs=embed, mask=attention_mask)
        return self.pooler(enc)
