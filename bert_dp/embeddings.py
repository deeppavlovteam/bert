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
                 sep_token_index: int = 103,
                 output_dim: int = 768,
                 use_one_hot_embedding: bool = False,  # currently is not used
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.sep_token_index = sep_token_index
        self.output_dim = output_dim
        self.max_len = max_len
        self.embeddings_initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

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
        self.token_matrix = self.add_weight(shape=(self.vocab_size, self.output_dim),
                                            initializer=self.embeddings_initializer,
                                            dtype=tf.float32,
                                            name='word_embeddings')

        self.segment_matrix = self.add_weight(shape=(2, self.output_dim),  # TODO: generalize to more than 2 segments
                                              initializer=self.embeddings_initializer,
                                              dtype=tf.float32,
                                              name='token_type_embeddings')
        self.pos_idxs = tf.stack([tf.range(self.max_len) for _ in range(batch_input_shape[0])])
        self.pos_matrix = self.add_weight(shape=(self.max_len, self.output_dim),
                                          initializer=self.embeddings_initializer,
                                          dtype=tf.float32,
                                          name='position_embeddings')

    def call(self,
             token_ids: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:

        pos_embeddings = embedding_ops.embedding_lookup(self.pos_matrix, self.pos_idxs[:, :token_ids.shape[1]])
        # TODO: replace lookup with matrix multiplication for small vocabs
        sep_ids = tf.cast(tf.equal(token_ids, self.sep_token_index), dtype=tf.int32)
        segment_ids = tf.cumsum(sep_ids, axis=1) - sep_ids
        segment_embeddings = embedding_ops.embedding_lookup(self.segment_matrix, segment_ids)
        token_embeddings = embedding_ops.embedding_lookup(self.token_matrix, token_ids)

        return pos_embeddings + segment_embeddings + token_embeddings
