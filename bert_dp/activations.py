import tensorflow as tf


def gelu(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Gaussian Error Linear Unit.

    This is a smoother version of the RELU. Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.math.sqrt(2.0)))
    return input_tensor * cdf
