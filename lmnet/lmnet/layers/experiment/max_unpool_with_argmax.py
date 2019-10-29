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
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.function import Defun


def max_unpool_with_argmax(x, argmax, ksize, data_format='NHWC'):
    """LeapMind max unpool with argmax function using Defun.
       Computes a partial inverse of MaxPoolWithArgmax.
       This function returns max unpooled tensor.
       If data format is `NCHW`, pre-transpose and post-transpose would be added.
    
       The function name in graph is `io.leapmind.MaxUnpoolWithArgmax`.

    Args:
        x (tf.Variable): The max pooled inputs.
        argmax (tf.Variable): The argmax from tf.nn.max_pool_with_argmax which has same
            shape with inputs.
        ksize (list): Size of the max pooling window.
        data_format: An optional string from "NHWC", "NCHW". Defaults to "NHWC".(Default value = 'NHWC')

    Returns:
        tf.Variable: An unpooled tensor.

    """

    def _backward(op, grad_quantized):
        """Backward.

        Args:
            op (tf.Operation): The forward operation.
            grad_quantized (tf.Tensor): The gradient w.r.t quantized input, weights normally.

        Returns:
            tf.Variable: The gradient w.r.t. normal (non-quantized) input.

        """
        flatten_grad = tf.reshape(grad_quantized, (grad_quantized.get_shape()[0], -1))
        flatten_argmax = tf.reshape(op.inputs[1], (op.inputs[1].get_shape()[0], -1))
        rs_list = []
        for i in range(grad_quantized.get_shape()[0]):
            rs_list.append(tf.gather(flatten_grad[i], flatten_argmax[i]))
        rs = tf.concat(rs_list, axis=0)
        return tf.reshape(rs, op.inputs[0].get_shape()), None, None, None

    def _forward(x, argmax, ksize):

        @Defun(tf.float32, tf.int64, tf.int32, tf.int32,
               python_grad_func=_backward,
               shape_func=lambda op: [np.multiply(op.inputs[0].get_shape().as_list(), ksize)],
               func_name="io.leapmind.MaxUnpoolWithArgmax")
        def _func(
                inputs,
                argmax,
                inputs_shape,
                ksize
        ):
            """Ref: https://github.com/mshunshin/SegNetCMR/blob/master/SegNetCMR/layers.py

            Args:
                inputs (tf.Variable): The max pooled inputs.
                argmax (tf.Variable): The argmax from tf.nn.max_pool_with_argmax which
                    has same shape with inputs.
                inputs_shape (list): Shape of inputs.
                ksize (list, tuple): The size of the window for each dimension of the
                    inputs.

            Returns:
                tf.Variable: An unpooled tensor.

            """
            # only int32 is registered to grad of FloorMod in GPU
            argmax = tf.cast(argmax, dtype=tf.int32)
            # calculate new shape
            output_shape = (inputs_shape[0], inputs_shape[1] * ksize[1], inputs_shape[2] * ksize[2], inputs_shape[3])
            # calculate indices for batch, height, width and feature maps
            one_like_argmax = tf.ones_like(argmax)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32),
                                     shape=tf.concat(([output_shape[0]], tf.ones((3,), dtype=tf.int32)), axis=0))
            b = one_like_argmax * batch_range
            y = argmax // (output_shape[2] * output_shape[3])
            x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
            channel_range = tf.range(output_shape[3], dtype=tf.int32)
            c = one_like_argmax * channel_range
            # transpose indices & reshape update values to one dimension
            inputs_size = tf.size(inputs)

            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, inputs_size]))

            values = tf.reshape(inputs, [inputs_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

        return _func(x, argmax, x.get_shape().as_list(), ksize)

    if data_format == "NHWC":
        return _forward(x, argmax, ksize)
    else:  # NCHW
        x = tf.transpose(x, perm=(0, 2, 3, 1))
        unpooled = _forward(x, argmax, ksize)
        return tf.transpose(unpooled, perm=(0, 3, 1, 2))
