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


def linear_mid_tread_half_quantizer(
        bit=None,
        max_value=None,
        backward=None,
        dtype=tf.float32,
):
    r"""Linear mid tread half quantizer.

        This quantization creates a linear mid tread half quantizer.
        If `backward` is provided, this `backward` will be used in backpropagation.

        This quantization method is DoReFa-Net [1]_ activation quantization variant, the differencce from DoReFa-Net [1]_ is to be able to change `max_value`.

        `op_type` is ``QTZ_linear_mid_tread_half``.

        Forward is:

        .. math::
            \mathbf{X} & = \text{clip}\big(\mathbf{X}, 0, max\_value\big)\\
            \mathbf{Y} & =
                \begin{cases}
                \mathbf{X},  & \text{if $bit$ is 32} \\
                \frac{\text{round}\big(\frac{\mathbf{X}}{max\_value}
                    \cdot (2^{bit}-1)\big)}{2^{bit}-1} \cdot max\_value, & otherwise
                \end{cases}

        Default backward is:

        .. math::
            \frac{\partial Loss}{\partial \mathbf{X}} =
                \begin{cases}
                \frac{\partial Loss}{\partial y},  & \text{if $0 < x < max\_value$}\\
                0, & otherwise
                \end{cases}

        Args:
            bit (int): Specify the bit of quantization.
            max_value (int): Be used for shift and clip.
            backward (callable): Be used in backpropagation.
            dtype (tf.DType): Define the data type of args of forward and backward.
        Returns:
            callable: forward function (grad_func defined).

        Reference:
            - `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_
            .. [1] `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_

    """  # NOQA
    min_value = 0

    def _backward(op, grad_quantized):
        """Backward.

        Args:
            op(tf.Operation): The forward operation.
            grad_quantized(tf.Tensor): The gradient w.r.t quantized input, weights normally.
        Returns:
            tf.Variable: The gradient w.r.t. normal (non-quantized) input.
        """
        if backward:
            return backward(op, grad_quantized)
        x = op.inputs[0]
        true = tf.ones(tf.shape(x))
        false = tf.zeros(tf.shape(x))
        dx = tf.where((x < max_value) & (x > min_value), true, false)
        return grad_quantized * dx, None, None

    def _forward(x):
        """Forward.

        Args:
            x(tf.Variable): The input to be quantized, weights normally.
        Returns:
            callable: The wrapped function which returns the quantized input.
        """

        @Defun(dtype, tf.int32, tf.float32, python_grad_func=_backward,
               shape_func=lambda op: [op.inputs[0].get_shape()],
               func_name='QTZ_linear_mid_tread_half')
        def _func(x, bit, max_value):
            n = tf.pow(2., tf.cast(bit, dtype=tf.float32)) - 1
            value_range = max_value - min_value

            x = tf.clip_by_value(x, min_value, max_value, name="clip")
            shifted = (x - min_value) / value_range
            quantized = tf.floor(shifted * n + 0.5) / n
            unshifted = quantized * value_range + min_value
            return unshifted

        return _func(x, bit, float(max_value))

    return _forward
