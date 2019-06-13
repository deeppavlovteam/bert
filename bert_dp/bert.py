from typing import Optional, Union

import tensorflow as tf
from tensorflow.python.keras.engine.network import Network

from .embeddings import AdvancedEmbedding
from .attention import MultiHeadSelfAttention
from .activations import gelu

# from .normalization import LayerNormalization
try:
    from tensorflow.keras.layers import LayerNormalization
except ImportError:
    from .normalization import LayerNormalization


class BERT(Network):
    """BERT body (implemented as a Network subclass in order to have weight-(de)serialization methods."""
    def __init__(self,
                 return_stack: Optional[bool] = None,
                 vocab_size: int = 119547,
                 token_type_vocab_size: int = 2,
                 sep_token_index: int = 102,
                 pad_token_index: int = 0,
                 emb_dropout_rate: float = 0.1,  # equal to hidden_dropout_prob in the original implementation
                 max_len: int = 512,
                 use_one_hot_embedding: bool = False,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,
                 layer_norm_epsilon: float = 1e-12,
                 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 pooler_fc_size: int = 768,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.return_stack = return_stack
        self.pad_token_index = pad_token_index

        # use name scopes for compatibility with both eager and graph modes
        with tf.name_scope('embeddings'):
            self.embed = AdvancedEmbedding(vocab_size=vocab_size,
                                           token_type_vocab_size=token_type_vocab_size,
                                           sep_token_index=sep_token_index,
                                           output_dim=hidden_size,
                                           use_one_hot_embedding=use_one_hot_embedding,
                                           max_len=max_len,
                                           initializer_range=initializer_range,
                                           trainable_pos_embedding=trainable_pos_embedding,
                                           trainable=trainable,
                                           name='embeddings')
            self.embed_dropout = tf.keras.layers.Dropout(rate=emb_dropout_rate,
                                                         trainable=trainable,
                                                         name='embeddings/dropout')
            self.embed_layer_norm = LayerNormalization(epsilon=layer_norm_epsilon,
                                                       trainable=trainable,
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

        self.pooler = tf.keras.layers.Dense(pooler_fc_size, activation='tanh', trainable=trainable, name='pooler/dense')

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        emb = self.embed(inputs, training=training)
        emb_norm_do = self.embed_dropout(self.embed_layer_norm(emb), training=training)

        # always compute mask if it is not provided
        if mask is None:
            mask = tf.cast(tf.not_equal(inputs, self.pad_token_index), tf.int32)
        enc = self.encoder(emb_norm_do, training=training, mask=mask)
        # For classification tasks, the first vector (corresponding to [CLS]) is used as the "sentence vector". Note
        # that this only makes sense because the entire model is fine-tuned.
        po = self.pooler(tf.squeeze(enc[:, 0:1, :], axis=1))

        if self.return_stack is None:
            return po
        elif self.return_stack:
            raise NotImplementedError('Currently all encoder layers output could not be obtained. You can get '
                                      'sequence output from the last encoder layer setting return_stack to False')
        else:
            return enc


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_heads: int = 12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.mhsa = MultiHeadSelfAttention(hidden_size=hidden_size,
                                           num_heads=num_heads,
                                           attention_probs_dropout_prob=attention_probs_dropout_prob,
                                           trainable=trainable,
                                           name='attention')

        self.dense = tf.keras.layers.Dense(units=hidden_size, name='attention/output/dense')

        self.dropout1 = tf.keras.layers.Dropout(rate=hidden_dropout_prob, name='attention/output/dropout')
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon, name='attention/output/LayerNorm')

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

        attn_output = self.mhsa(inputs, mask=mask)
        attn_output = self.dropout1(self.dense(attn_output), training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.pff2(self.pff1(out1))
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        out2._keras_mask = mask  # workaround for mask propagation  # TODO: investigate self.supports_masking usage
        return out2

    def compute_mask(self,
                     inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape
