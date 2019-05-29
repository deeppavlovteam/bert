from typing import Optional

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops


class BERTCombinedEmbedding(tf.keras.layers.Layer):
    """Embed token_type_ids, position_ids and token_ids and return the sum."""
    def __init__(self,
                 vocab_size: int = 119547,
                 token_type_vocab_size: int = 2,
                 sep_token_index: int = 102,
                 output_dim: int = 768,
                 use_one_hot_embedding: bool = False,  # currently is not used
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.token_type_vocab_size = token_type_vocab_size
        self.sep_token_index = sep_token_index
        self.output_dim = output_dim
        self.max_len = max_len
        self.embeddings_initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
        self.trainable_pos_embedding = trainable_pos_embedding

    @tf_utils.shape_type_conversion
    def build(self, batch_input_shape):
        # Note: most sparse optimizers do not have GPU kernels defined. When
        # building graphs, the placement algorithm is able to place variables on CPU
        # since it knows all kernels using the variable only exist on CPU.
        # When eager execution is enabled, the placement decision has to be made
        # right now. Checking for the presence of GPUs to avoid complicating the
        # TPU codepaths which can handle sparse optimizers.
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self._create_weights(batch_input_shape)
        else:

            self._create_weights(batch_input_shape)
        self.built = True

    def _create_weights(self, batch_input_shape):
        self.token_emb_table = self.add_weight(shape=(self.vocab_size, self.output_dim),
                                               dtype=tf.float32,
                                               initializer=self.embeddings_initializer,
                                               name='word_embeddings')

        self.token_type_emb_table = self.add_weight(shape=(self.token_type_vocab_size, self.output_dim),
                                                    dtype=tf.float32,
                                                    initializer=self.embeddings_initializer,
                                                    name='token_type_embeddings')
        self.full_position_emb_table = self.add_weight(shape=(self.max_len, self.output_dim),
                                                       dtype=tf.float32,
                                                       initializer=self.embeddings_initializer,
                                                       trainable=self.trainable_pos_embedding,
                                                       name='position_embeddings')

    def call(self,
             token_ids: tf.Tensor,
             training: Optional[bool] = None,
             # mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:

        # TODO: maybe add type checking
        token_emb = embedding_ops.embedding_lookup(self.token_emb_table, tf.cast(token_ids, tf.int32))

        # pos_emb = tf.slice(self.full_position_emb_table,
        #                    begin=[0, 0], size=[token_ids.shape.as_list()[1], -1])
        # TODO: try to generalize to dynamic sequence length dimension
        pos_emb = self.full_position_emb_table[:token_ids.shape[1], :]

        sep_ids = tf.cast(tf.equal(token_ids, self.sep_token_index), dtype=tf.int32)
        segment_ids = tf.cumsum(sep_ids, axis=1) - sep_ids

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_segment_ids = tf.reshape(segment_ids, [-1])
        oh_segment_ids = tf.one_hot(flat_segment_ids, depth=self.token_type_vocab_size)
        segment_emb = tf.matmul(oh_segment_ids, self.token_type_emb_table)
        # TODO: try to generalize to dynamic sequence length dimension
        segment_emb = tf.reshape(segment_emb, [-1] + token_emb.shape.as_list()[1:])

        return token_emb + pos_emb + segment_emb

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.max_len, self.output_dim
