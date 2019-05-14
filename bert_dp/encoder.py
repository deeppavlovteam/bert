from typing import Optional, Union

import tensorflow as tf

from .activations import gelu
from .attention import MultiHeadAttention
from .normalization import LayerNormalization


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.layers = []
        for i in range(num_hidden_layers):
            self.layers.append(TransformerBlock(hidden_size=hidden_size,
                                                intermediate_size=intermediate_size,
                                                num_heads=num_heads,
                                                dropout_rate=dropout_rate,
                                                intermediate_act_fn=intermediate_act_fn,
                                                layer_norm_epsilon=layer_norm_epsilon,
                                                name=f'layer_{i}'))

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:
        x = inputs
        for l in self.layers:
            x = l(x, mask=mask)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_heads: int = 12,
                 dropout_rate: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, name='attention')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon,
                                             name='attention/output/LayerNorm')

        # point-wise feed forward network
        self.pff1 = tf.keras.layers.Dense(intermediate_size,
                                          activation=intermediate_act_fn,
                                          name='intermediate/dense')
        self.pff2 = tf.keras.layers.Dense(hidden_size, activation='relu', name='output/dense')

        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon, name='output/LayerNorm')

    def call(self, inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:
        x = inputs
        attn_output, _ = self.mha({'query': x, 'key': x, 'value': x}, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.pff2(self.pff1(out1))  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
