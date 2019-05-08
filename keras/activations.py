import tensorflow as tf


def gelu(input_tensor):
    """
    Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


# def not_accurate_gelu(x):
#     return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
#
#
# class Gelu(tf.keras.Layer):
#     def __init__(self, accurate: bool = False, **kwargs):
#         super().__init__(**kwargs)
#         self.accurate = accurate
#
#     def call(self, inputs, **kwargs):
#         if not self.accurate:
#             return not_accurate_gelu(inputs)
#         # if K.backend() == 'tensorflow':
#         #     erf = K.tf.erf
#         # else:
#         #     erf = K.T.erf
#         return inputs * 0.5 * (1.0 + tf.math.erf(inputs / tf.math.sqrt(2.0)))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         config = {
#             'accurate': self.accurate,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))
