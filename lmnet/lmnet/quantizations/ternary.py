# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from tensorflow.python.framework.function import Defun

# TODO(wakisaka): Write math doc.
# TODO(wakisaka): Decide operation name.


def ttq_weight_quantizer(threshold=0.05, dtype=tf.float32):
    """Trained Ternary Quantization (TTQ)

    Ref: https://arxiv.org/pdf/1612.01064v1.pdf
    """

    @Defun(dtype, dtype, dtype, tf.float32, dtype)
    def backward(weights, positive, negative, threshold, grad_quantized):
        """Backward
        Args:
            weights(tf.Variable): The weights to be quantized.
            positive(tf.Variable): The positive scaling factor.
            negative(tf.Variable): The negative scaling factor.
            threshold(tf.Tensor): The threshold scalar value.
            grad_quantized(tf.Tensor): The gradient w.r.t quantized weights.
        Return:
            grad_weights(tf.Tensor): The gradient w.r.t. normal (non-quantized) weights.
            grad_positive(tf.Tensor): The gradient w.r.t. positive.
            grad_negative(tf.Tensor): The gradient w.r.t. negative.
            grad_threshold(tf.Tensor): The gradient w.r.t. threshold. It be ignore because of the threshold is constant value.
        """ # NOQA
        ternary_threshold = tf.reduce_max(tf.abs(weights)) * threshold
        mask_positive = (weights > ternary_threshold)
        mask_negative = (weights < -ternary_threshold)
        mask_middle = ~mask_positive & ~mask_negative

        grad_positive = tf.reduce_sum(tf.where(mask_positive, grad_quantized, tf.zeros_like(weights)))
        grad_negative = tf.reduce_sum(tf.where(mask_negative, grad_quantized, tf.zeros_like(weights)))

        positive_grad_weights = tf.where(mask_positive, grad_quantized * positive, tf.zeros_like(weights))
        negative_grad_weights = tf.where(mask_negative, grad_quantized * negative, tf.zeros_like(weights))
        middle_grad_weights = tf.where(mask_middle, grad_quantized, tf.zeros_like(weights))

        grad_weights = positive_grad_weights + negative_grad_weights + middle_grad_weights

        grad_threshold = tf.constant(0.0)
        return grad_weights, grad_positive, grad_negative, grad_threshold

    @Defun(dtype, dtype, dtype, tf.float32,
           grad_func=backward, func_name="ttq_weight_quantizer", shape_func=lambda op: [op.inputs[0].get_shape()])
    def forward(weights, positive, negative, threshold):
        """Forward
        Args:
            weights(tf.Variable): The weights to be quantized.
            positive(tf.Variable): The positive scaling factor.
            negative(tf.Variable): The negative scaling factor.
            threshold(tf.Tensor): The threshold scalar value.
        Returns:
            quantized(tf.Variable): The quantized weights.
        """
        ternary_threshold = tf.reduce_max(tf.abs(weights)) * threshold
        mask_positive = (weights > ternary_threshold)
        mask_negative = (weights < -ternary_threshold)

        positive_weights = tf.where(mask_positive, tf.ones_like(weights) * positive, tf.zeros_like(weights))
        negative_weights = - tf.where(mask_negative, tf.ones_like(weights) * negative, tf.zeros_like(weights))
        quantized = positive_weights + negative_weights
        return quantized

    def wrapper(weights):
        """Initialize positive, negative variables as call"""

        # TODO(wakisaka): These vars should be constrain to be more than 0.
        # tf.get_variable's constrain argument needs tensorflow v1.4.
        # See https://stackoverflow.com/questions/33694368/what-is-the-best-way-to-implement-weight-constraints-in-tensorflow/37426800  # NOQA
        positive = tf.get_variable("positive", initializer=1.0)
        negative = tf.get_variable("negative", initializer=1.0)
        tf.summary.scalar("positive", positive)
        tf.summary.scalar("negative", negative)
        return forward(weights, positive, negative, threshold)

    return wrapper


def twn_weight_quantizer(threshold=0.7, dtype=tf.float32):
    """Ternary Weight Networks (TWN)

    Ref: https://arxiv.org/abs/1605.04711
    """

    @Defun(dtype, dtype)
    def backward(weights, grad_quantized):
        """Backward
        Args:
            grad_quantized(tf.Tensor): The gradient w.r.t quantized weights.
        Return:
            grad_weights(tf.Tensor): The gradient w.r.t. normal (non-quantized) weights.
        """
        grad_weights = grad_quantized
        return grad_weights

    @Defun(dtype, grad_func=backward, func_name="twn_weight_quantizer",
           shape_func=lambda op: [op.inputs[0].get_shape()])
    def forward(weights):
        """Forward
        Args:
            weights(tf.Variable): The weights to be quantized.
        Returns:
            quantized(tf.Variable): The quantized weights.
        """
        ternary_threshold = tf.reduce_sum(tf.abs(weights)) * threshold / tf.cast(tf.size(weights), tf.float32)
        mask_positive = (weights > ternary_threshold)
        mask_negative = (weights < -ternary_threshold)
        mask_p_or_n = mask_positive | mask_negative

        p_or_n_weights = tf.where(mask_p_or_n, weights, tf.zeros_like(weights))
        scaling_factor = tf.reduce_sum(tf.abs(p_or_n_weights)) / tf.reduce_sum(tf.cast(mask_p_or_n, tf.float32))

        positive_weights = scaling_factor * tf.where(mask_positive, tf.ones_like(weights), tf.zeros_like(weights))
        negative_weights = - scaling_factor * tf.where(mask_negative, tf.ones_like(weights), tf.zeros_like(weights))

        quantized = positive_weights + negative_weights
        return quantized

    return forward
