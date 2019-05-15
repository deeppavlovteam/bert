import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 epsilon=1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = tf.constant(epsilon, dtype=tf.float32)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)
        super().build(input_shape)

    def call(self,
             x,
             trainable=None,
             mask=None,
             **kwargs):
        u = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.math.reduce_mean(tf.math.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / tf.math.sqrt(s + self.eps)
        return self.gamma * z + self.beta
