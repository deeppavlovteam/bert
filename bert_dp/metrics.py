import types

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.metrics import Metric


class BinaryF1Score(Metric):

    def __new__(cls, *args, **kwargs):
        # don't look at this
        obj = super(Metric, cls).__new__(cls)
        update_state_fn = obj.update_state
        obj.update_state = types.MethodType(metrics_utils.update_state_wrapper(update_state_fn), obj)
        obj.result = types.MethodType(metrics_utils.result_wrapper(obj.result), obj)
        return obj

    def __init__(self):

        super(BinaryF1Score, self).__init__()

        default_threshold = 0.5
        self.thresholds = metrics_utils.parse_init_thresholds(None, default_threshold=default_threshold)
        self.tp = self.add_weight('true_positives',
                                  shape=(1,),
                                  initializer=init_ops.zeros_initializer)
        self.fp = self.add_weight('false_positives',
                                  shape=(1,),
                                  initializer=init_ops.zeros_initializer)
        self.fn = self.add_weight('false_negatives',
                                  shape=(1,),
                                  initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.fp,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.fn
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=None,
            class_id=None,
            sample_weight=sample_weight)

    def result(self):
        result = math_ops.div_no_nan(2 * self.tp, (2 * self.tp + self.fp + self.fn))
        return result[0]

    def reset_states(self):
        tf.keras.backend.batch_set_value([(v, np.zeros((1,))) for v in self.variables])
