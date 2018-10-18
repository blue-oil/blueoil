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


def binary_channel_wise_mean_scaling_quantizer(
        backward=None,
        dtype=tf.float32,
):
    r"""Binary channel wise mean scaling quantizer.

        This quantization creates a binary channel wise mean scaling quantizer.
        If `backward` is provided, this `backward` will be used in backpropagation.

        This method is varient of XNOR-Net [1]_ weight quantization, the differencce from XNOR-Net [1]_ is backward function.

        `op_type` is ``QTZ_binary_channel_wise_mean_scaling``.

        Forward is:

        .. math::
            \begin{align}
                \bar{\mathbf{x}} & = \frac{1}{n}||\mathbf{X}||_{\ell1}
                & \text{$\bar{\mathbf{x}}$ is a $c$-channels vector} \\
                & & \text{$n$ is number of elements in each channel of $\mathbf{X}$} \\\\
                \mathbf{Y} & = \text{sign}\big(\mathbf{X}\big) \times \bar{\mathbf{x}} &\\
            \end{align}

        Default backward is:

        .. math::
            \frac{\partial Loss}{\partial \mathbf{X}} = \frac{\partial Loss}{\partial \mathbf{Y}}

        Args:
            backward (callable): Be used in backpropagation.
            dtype (tf.DType): Define the data type of args of forward and backward.
        Returns:
            callable: forward function (grad_func defined).

        Reference:
            .. [1] `XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks <https://arxiv.org/abs/1603.05279>`_
    """  # NOQA

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
        return grad_quantized

    @Defun(dtype, python_grad_func=_backward, shape_func=lambda op: [op.inputs[0].get_shape()],
           func_name='QTZ_binary_channel_wise_mean_scaling')
    def _forward(x):
        """Forward.

        Args:
            x(tf.Variable): The input to be quantized, weights normally.
        Returns:
            tf.Variable: The quantized input.
        """
        # x kernel shape is [height, width, in_channels, out_channels]
        scaling_factor = tf.reduce_mean(tf.abs(x), axis=[0, 1, 2])
        # TODO(wakisaka): tensorflow raise error.
        # tf.summary.histogram("scaling_factor", scaling_factor)
        quantized = tf.sign(x) * scaling_factor
        return quantized

    return _forward


def binary_mean_scaling_quantizer(
        backward=None,
        dtype=tf.float32,
):
    r"""Binary mean scaling quantizer.

        This quantization creates a binary mean scaling quantizer.
        If `backward` is provided, this `backward` will be used in backpropagation.

        This method is DoReFa-Net [2]_ weight quantization.

        `op_type` is ``QTZ_binary_mean_scaling``.

        Forward is:

        .. math::
            \begin{align}
                \bar{x} & = \frac{1}{N}||\mathbf{X}||_{\ell1}
                & \text{$\bar{x}$ is a scalar} \\
                & & \text{$N$ is number of elements in all channels of $\mathbf{X}$}\\
                \mathbf{Y} & = \text{sign}\big(\mathbf{X}\big) \cdot \bar{x} &\\
            \end{align}

        Default backward is:

        .. math::
            \frac{\partial Loss}{\partial \mathbf{X}} = \frac{\partial Loss}{\partial \mathbf{Y}}

        Args:
            backward (callable): Be used in backpropagation.
            dtype (tf.DType): Define the data type of args of forward and backward.
        Returns:
            callable: forward function (grad_func defined).

        Reference:
             .. [2] `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_
    """ # NOQA

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
        return grad_quantized

    @Defun(dtype, python_grad_func=_backward, shape_func=lambda op: [op.inputs[0].get_shape()],
           func_name='QTZ_binary_mean_scaling')
    def _forward(x):
        """Forward.

        Args:
            x(tf.Variable): The input to be quantized, weights normally.
        Returns:
            tf.Variable: The quantized input.
        """
        expectation = tf.reduce_mean(tf.abs(x))
        return tf.sign(x) * expectation

    return _forward
