from typing import Optional

import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    """Layer that normalize outputs to statistics aggregated across hidden dimension"""
    def __init__(self,
                 epsilon=1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.eps = tf.constant(epsilon, dtype=tf.float32)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.keras.initializers.Zeros())
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones())
        super().build(input_shape)

    def call(self,
             x: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> None:
        u = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.math.reduce_mean(tf.math.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / tf.math.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_mask(self,
                     inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return mask

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
