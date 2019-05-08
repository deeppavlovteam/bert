import tensorflow as tf
import numpy as np

# def layer_norm(input_tensor, name=None):
#   """Run layer normalization on the last dimension of the tensor."""
#   return tf.contrib.layers.layer_norm(
#       inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def _get_pos_encoding_matrix(max_len: int, d_emb: int) -> np.array:
    pos_enc = np.array(
        [[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] if pos != 0 else np.zeros(d_emb) for pos in
         range(max_len)], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class BERTInputEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 output_dim: int = 768,
                 use_one_hot_embedding: bool = False,  # currently is not used
                 dropout: float = 0.1,
                 vocab_size: int = 119547,
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
                 layer_norm_epsilon: float = 1e-12,
                 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315
                 # is tf.keras version identical to the original layer_norm, which is copied above?
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_len = max_len
        # self.use_one_hot_embedding = use_one_hot_embedding
        self.output_dim = output_dim
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.trainable_pos_embedding = trainable_pos_embedding

        self.segment_emb = tf.keras.layers.Embedding(input_dim=2,  # either 0 or 1 segment id
                                                     output_dim=output_dim,
                                                     embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                         stddev=initializer_range),
                                                     input_length=max_len,
                                                     name='token_type_embeddings')

        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len,
                                                 output_dim=output_dim,
                                                 embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                     stddev=initializer_range),
                                                 trainable=trainable_pos_embedding,
                                                 input_length=max_len,
                                                 name='position_embeddings',
                                                 weights=None if trainable_pos_embedding else [_get_pos_encoding_matrix(
                                                     max_len, output_dim)])

        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=output_dim,
                                                   embeddings_initializer=tf.keras.initializers.TruncatedNormal(
                                                       stddev=initializer_range),
                                                   input_length=max_len,
                                                   name='word_embeddings')
        self.embedding_dropout = tf.keras.layers.Dropout(rate=dropout)
        self.embedding_layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon, name='LayerNorm')

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim

    def call(self, inputs, training=None, mask=None, **kwargs):
        token_ids, segment_ids, pos_ids = inputs
        segment_embeddings = self.segment_emb(segment_ids)
        pos_embeddings = self.pos_emb(pos_ids)
        token_embeddings = self.token_emb(token_ids)
        emb_sum = segment_embeddings + pos_embeddings + token_embeddings
        emb_sum_norm = self.embedding_layer_norm(emb_sum)
        return self.embedding_dropout(emb_sum_norm)

    # def get_config(self):
    #     config = {
    #         'max_len': self.max_len,
    #         'use_one_dropout': self.use_one_dropout,
    #         'output_dim': self.output_dim,
    #         'dropout': self.dropout,
    #         'vocab_size': self.vocab_size,
    #         'trainable_pos_embedding': self.trainable_pos_embedding,
    #         'embedding_layer_norm': self.use_embedding_layer_norm,
    #         'layer_norm_epsilon': self.layer_norm_epsilon
    #     }
    #     base_config = super().get_config()
    #     return dict(list(base_config.items()) + list(config.items()))
