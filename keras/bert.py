from typing import Collection, Optional

import tensorflow as tf
from .embeddings import BERTInputEmbedding
from .encoder import TransformerEncoder


class BERT(tf.keras.layers.Layer):
    """BERT body"""
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 pooler_fc_size: int = 768,
                 **kwargs) -> None:
        super().__init__(name='bert', **kwargs)
        self.emb = BERTInputEmbedding()
        self.encoder = TransformerEncoder(hidden_size=hidden_size,
                                          intermediate_size=intermediate_size,
                                          num_hidden_layers=num_hidden_layers)
        self.pooler = tf.keras.layers.Dense(pooler_fc_size, name='pooler')

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
             inputs: Collection[tf.Tensor],
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:
        # token_ids, segment_ids, pos_ids = inputs
        embed = self.emb(inputs)
        attention_mask = self.create_self_attention_mask_from_input_mask(mask)
        enc = self.encoder(inputs=embed, mask=attention_mask)
        return self.pooler(enc)
