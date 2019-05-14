from typing import Mapping, Optional

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops


class BERTCombinedEmbedding(tf.keras.layers.Layer):
    """Embed token_type_ids, position_ids and token_ids and return the sum."""
    def __init__(self,
                 output_dim: int = 768,
                 use_one_hot_embedding: bool = False,  # currently is not used
                 vocab_size: int = 119547,
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,  # always True in the original implementation
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embeddings_initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Note: most sparse optimizers do not have GPU kernels defined. When
        # building graphs, the placement algorithm is able to place variables on CPU
        # since it knows all kernels using the variable only exist on CPU.
        # When eager execution is enabled, the placement decision has to be made
        # right now. Checking for the presence of GPUs to avoid complicating the
        # TPU codepaths which can handle sparse optimizers.
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self._create_weights(input_shape)
        else:
            self._create_weights(input_shape)
        self.built = True

    def _create_weights(self, input_shape):
        self.segment_matrix = self.add_weight(shape=(2, self.output_dim),
                                              initializer=self.embeddings_initializer,
                                              dtype=tf.float32,
                                              name='token_type_embeddings')
        self.pos_matrix = self.add_weight(shape=(self.max_len, self.output_dim),
                                          initializer=self.embeddings_initializer,
                                          dtype=tf.float32,
                                          name='position_embeddings')
        self.token_matrix = self.add_weight(shape=(self.vocab_size, self.output_dim),
                                            initializer=self.embeddings_initializer,
                                            dtype=tf.float32,
                                            name='word_embeddings')

    def call(self,
             segment_ids,
             pos_ids,
             token_ids,
             # inputs: Mapping[str, tf.Tensor],
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:

        # TODO: replace lookup with matrix multiplication for small vocabs
        segment_embeddings = embedding_ops.embedding_lookup(self.segment_matrix, segment_ids)  # inputs['segment_ids'])
        pos_embeddings = embedding_ops.embedding_lookup(self.pos_matrix, pos_ids)  # inputs['pos_ids'])
        token_embeddings = embedding_ops.embedding_lookup(self.token_matrix, token_ids)  # inputs['token_ids'])

        return segment_embeddings + pos_embeddings + token_embeddings
