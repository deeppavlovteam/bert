from typing import Union, Optional, Collection

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import math_ops, control_flow_ops


class AdamW(optimizer_v2.OptimizerV2):
    """
    ### Write a customized optimizer.
    If you intend to create your own optimization algorithm, simply inherit from
    this class and override the following methods:
    - resource_apply_dense (update variable given gradient tensor is dense)
    - resource_apply_sparse (update variable given gradient tensor is sparse)
    - create_slots (if your optimizer algorithm requires additional variables)
    - get_config (serialization of the optimizer, include all hyper parameters)
    """
    def __init__(self,
                 learning_rate: Union[float, callable],
                 weight_decay_rate: float = 0.0,  # self._initial_decay ?
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon:float = 1e-6,
                 exclude_from_weight_decay: Optional[Collection] = None,
                 name: str = "AdamW",
                 **kwargs) -> None:
        super(AdamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self.weight_decay_rate = weight_decay_rate  # ?
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay  # ?

    def _resource_apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _resource_apply_sparse(self, grad, var, indices):
        pass

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for v in var_list:
            self.add_slot(var=v, slot_name='m', initializer='zeros')
        for v in var_list:
            self.add_slot(var=v, slot_name='v', initializer='zeros')

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),  # weight_decay_rate?
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'exclude_from_weight_decay': self.exclude_from_weight_decay,
        })
        return config
