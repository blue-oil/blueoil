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
"""Exporter from DLK to TensorFlow."""

import functools
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.function import Defun

from core import model as dlk
from core.data_types import DataType
from core.graph import Graph as dlk_Graph
from core.graph import GraphRunner
from core.operators import Add, AveragePool, BatchNormalization, Constant, Conv, Identity, Input, \
    MaxPool, Operator, Output, Transpose, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half, \
    Reshape, Softmax, Relu, Flatten, Dropout, Gemm

TF_DTYPE_MAP: Dict[str, tf.DType] = {
    'Float16': tf.float16,
    'Float32': tf.float32,
    'Float64': tf.float64,
    'Uint8': tf.uint8,
    'Uint16': tf.uint16,
    'Uint32': None,
    'Uint64': None,
    'Int8': tf.int8,
    'Int16': tf.int16,
    'Int32': tf.int32,
    'Int64': tf.int64,

    'Bool': tf.bool,
    'String': tf.string,
}


class Exporter(GraphRunner):

    @classmethod
    def export_graph(cls, model: dlk.Model) -> tf.Graph:
        dlk_graph = model.graph

        runner = cls(dlk_graph)
        runner.run()

        return runner.tf_graph

    def __init__(self, graph: dlk_Graph) -> None:
        self._tf_graph = tf.Graph()
        self.tf_ops: Dict[str, tf.Tensor] = {}
        self._formats: Dict[str, str] = {}
        self._permutation: Dict[str, List[int]] = {}
        super().__init__(graph)

    @property
    def tf_graph(self) -> tf.Graph:
        return self._tf_graph

    # initialize and finalize

    def initialize(self, **kwargs: Any) -> None:
        """Set up TF's default graph"""
        # self._tf_graph.as_default().__enter__()

    def finalize(self, **kwargs: Any) -> None:
        """Release the TF default graph"""
        # self._tf_graph.as_default().__exit__(None, None, None)

    # backward run: check the data format and transpose if needed

    def _transpose_weights(self, node: Operator) -> Optional[str]:
        given_format = self._formats.get(node.name)
        set_format = node.dimension
        if given_format and len(given_format) == 4 and given_format != set_format:
            perm = [set_format.index(s) for s in given_format]
            self._permutation[node.name] = perm

        return given_format

    def run_backward_constant(self, node: Constant, **kwargs: Any) -> None:
        self._transpose_weights(node)

    def _transpose_if_not_supported(self, node: Operator) -> None:
        if node.dimension not in {'NHWC', 'NCHW'}:
            perm = [node.dimension.index(s) for s in 'NHWC']
            self._permutation[node.name] = perm

    def run_backward_identity(self, node: Identity, **kwargs: Any) -> None:
        given_format = self._transpose_weights(node)
        if given_format:
            self._formats[node.input_ops['input'].name] = given_format

    def run_backward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:
        given_format = self._transpose_weights(node)
        if given_format:
            self._formats[node.input_ops['input'].name] = given_format

    def run_backward_transpose(self, node: Transpose, **kwargs: Any) -> None:
        given_format = self._transpose_weights(node)
        if given_format:
            # change the input's format
            perm = node.permutation
            inv_perm = [perm[i] for i in range(len(perm))]  # inverse the perm
            transposed_form = functools.reduce(
                lambda x, y: x + y, [given_format[i] for i in inv_perm])
            self._formats[node.input_ops['data'].name] = transposed_form

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        # if the format is not supported, change their order
        self._transpose_if_not_supported(node)
        self._formats[node.input_ops['W'].name] = 'HWCN'

    def run_backward_batch_normalization(self, node: BatchNormalization, **kwargs: Any) -> None:
        given_format = self._transpose_weights(node)
        if given_format:
            self._formats[node.input_ops['X'].name] = given_format

    def run_backward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        given_format = self._transpose_weights(node)
        if given_format:
            self._formats[node.input_ops['X'].name] = given_format

    def run_backward_max_pool(self, node: MaxPool, **kwargs: Any) -> None:
        # if the format is not supported, change their order
        self._transpose_if_not_supported(node)

    def run_backward_average_pool(self, node: AveragePool, **kwargs: Any) -> None:
        # if the format is not supported, change their order
        self._transpose_if_not_supported(node)

    # forward run: create tf operators

    def _get_tf_dtype(self, dlk_dtype: DataType) -> tf.DType:
        dtype = TF_DTYPE_MAP.get(dlk_dtype.name())
        if dtype:
            return dtype
        else:
            raise ValueError(f'dtype {dlk_dtype.name} is not supported.')

    def _get_transposed_or_not(self, node: Operator):
        if node.name in self._permutation.keys():
            perm = self._permutation[node.name]
            new_shape: List[int] = [node.shape[i] for i in perm]
            new_dimension: str = functools.reduce(
                lambda x, y: x + y, [node.dimension[i] for i in perm])
            new_data: np.ndarray = node.data.transpose(perm)
            return new_shape, new_dimension, new_data
        else:
            return node.shape, node.dimension, node.data

    def run_forward_input(self, node: Input, **kwargs: Any) -> None:
        new_shape, _, _ = self._get_transposed_or_not(node)
        with self._tf_graph.as_default():
            x = tf.placeholder(self._get_tf_dtype(node.dtype), shape=new_shape, name=node.name)
        self.tf_ops[node.name] = x

    def run_forward_constant(self, node: Constant, **kwargs: Any) -> None:
        new_shape, _, new_data = self._get_transposed_or_not(node)
        with self._tf_graph.as_default():
            x = tf.constant(new_data, dtype=self._get_tf_dtype(node.dtype),
                            shape=new_shape,
                            name=node.name)
        self.tf_ops[node.name] = x

    def run_forward_output(self, node: Output, **kwargs: Any) -> None:
        input = self.tf_ops[node.input_ops['input'].name]
        with self._tf_graph.as_default():
            x = tf.identity(input, name=node.name)
        self.tf_ops[node.name] = x

    def run_forward_identity(self, node: Identity, **kwargs: Any) -> None:
        input = self.tf_ops[node.input_ops['input'].name]
        with self._tf_graph.as_default():
            x = tf.identity(input, name=node.name)
        self.tf_ops[node.name] = x

    def run_forward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:

        x = self.tf_ops[node.input_ops['input'].name]

        @Defun(self._get_tf_dtype(node.dtype), shape_func=lambda op: [op.inputs[0].get_shape()],
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

        with self._tf_graph.as_default():
            output = _forward(x, name=node.name)
        self.tf_ops[node.name] = output

    def run_forward_transpose(self, node: Transpose, **kwargs: Any) -> None:
        perm = node.permutation
        a = self.tf_ops[node.input_ops['data'].name]
        with self._tf_graph.as_default():
            x = tf.transpose(a, perm, name=node.name)
        self.tf_ops[node.name] = x

    def _get_padding2D(self, input_shape: List[int], kernel_shape: List[int]) -> str:
        return 'SAME' if input_shape == kernel_shape else 'VALID'

    def run_forward_conv(self, node: Conv, **kwargs: Any) -> None:
        if node.dilations != [1, 1, 1, 1]:
            ValueError(f'Tensorflow v1.4 does not support dilations {node.dilations}')

        x = self.tf_ops[node.input_ops['X'].name]
        w = self.tf_ops[node.input_ops['W'].name]

        inputs = [x, w]
        dtypes = [self._get_tf_dtype(node.dtype)]
        attrs: Dict[str, Any] = {}

        dim = node.dimension
        strides = [1, *(node.strides), 1] if dim == 'NHWC' \
            else [1, 1, *(node.strides)]  # dim == 'NCHW'
        in_x = node.input_ops['X']
        padding = self._get_padding2D([in_x.height, in_x.width], [node.height, node.width])

        with self._tf_graph.as_default():
            y = tf.nn.conv2d(x, w, strides, padding, name=node.name,
                             data_format=dim)
        self.tf_ops[node.name] = y

    def run_forward_batch_normalization(self, node: BatchNormalization, **kwargs: Any) -> None:
        x = self.tf_ops[node.input_ops['X'].name]
        scale = self.tf_ops[node.input_ops['scale'].name]
        b = self.tf_ops[node.input_ops['B'].name]
        mean = self.tf_ops[node.input_ops['mean'].name]
        var = self.tf_ops[node.input_ops['var'].name]
        epsilon = node.epsilon

        # param_initializer = {'beta': b, 'gamma': scale, 'moving_mean': mean, 'moving_variance': var}
        # test = tf.constant_initializer(10)

        with self._tf_graph.as_default():
            # b = tf.constant_initializer(b)
            # scale = tf.constant_initializer(scale)
            # mean = tf.constant_initializer(mean)
            # var = tf.constant_initializer(var)
            # y = tf.layers.batch_normalization(x, beta_initializer=b, gamma_initializer=scale,
            #                                   moving_mean_initializer=mean,
            #                                   moving_variance_initializer=var,
            #                                   epsilon=epsilon, fused=True)
            y = tf.nn.fused_batch_norm(x, scale, b, mean=mean, variance=var, epsilon=epsilon, is_training=False,
                                       name=node.name)
            # y = tf.nn.batch_normalization(x, mean, var, b, scale, epsilon, name=node.name)
            # y = tf.contrib.layers.batch_norm(x, center=True, scale=True, epsilon=epsilon, fused=True)
        self.tf_ops[node.name] = y[0]

    def run_forward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        x = self.tf_ops[node.input_ops['X'].name]
        bit = self.tf_ops[node.input_ops['Y'].name]
        max_value = self.tf_ops[node.input_ops['Z'].name]

        @Defun(self._get_tf_dtype(node.dtype), tf.int32, tf.float32,
               shape_func=lambda op: [op.inputs[0].get_shape()],
               func_name='QTZ_linear_mid_tread_half')
        def _func(x, bit, max_value):
            min_value = 0
            n = tf.pow(2., tf.cast(bit, dtype=tf.float32)) - 1
            value_range = max_value - min_value

            x = tf.clip_by_value(x, min_value, max_value, name="clip")
            shifted = (x - min_value) / value_range
            quantized = tf.round(shifted * n) / n
            unshifted = quantized * value_range + min_value
            return unshifted

        with self._tf_graph.as_default():
            output = _func(x, bit, max_value, name=node.name)
        self.tf_ops[node.name] = output

    def run_forward_add(self, node: Add, **kwargs: Any) -> None:
        x = self.tf_ops[node.input_ops['A'].name]
        y = self.tf_ops[node.input_ops['B'].name]

        with self._tf_graph.as_default():
            c = tf.add(x, y, name=node.name)
        self.tf_ops[node.name] = c

    def run_forward_max_pool(self, node: MaxPool, **kwargs: Any) -> None:
        x = self.tf_ops[node.input_ops['X'].name]
        ksize = [node.kernel_height, node.kernel_width]
        strides = node.strides
        in_x = node.input_ops['X']
        padding = self._get_padding2D([in_x.height, in_x.width], [node.height, node.width])

        with self._tf_graph.as_default():
            y = tf.nn.max_pool(x, ksize, strides, padding, name=node.name)
        self.tf_ops[node.name] = y

    def run_forward_average_pool(self, node: AveragePool, **kwargs: Any) -> None:
        x = self.tf_ops[node.input_ops['X'].name]
        ksize = [node.kernel_height, node.kernel_width]
        strides = node.strides
        in_x = node.input_ops['X']
        padding = self._get_padding2D([in_x.height, in_x.width], [node.height, node.width])

        y = tf.nn.avg_pool(x, ksize, strides, padding, name=node.name)
        self.tf_ops[node.name] = y

    def run_forward_reshape(self, node: Reshape, **kwargs: Any) -> None:
        tensor = self.tf_ops[node.input_ops['data'].name]
        shape = node.shape

        with self._tf_graph.as_default():
            reshaped = tf.reshape(tensor, shape, name=node.name)
        self.tf_ops[node.name] = reshaped

    def run_forward_softmax(self, node: Softmax, **kwargs: Any) -> None:
        logits = self.tf_ops[node.input_ops['input'].name]

        with self._tf_graph.as_default():
            output = tf.nn.softmax(logits, name=node.name)
        self.tf_ops[node.name] = output

    def run_forward_relu(self, node: Relu, **kwargs: Any) -> None:
        features = self.tf_ops[node.input_ops['X'].name]

        with self._tf_graph.as_default():
            y = tf.nn.relu(features, name=node.name)
        self.tf_ops[node.name] = y

    def run_forward_flatten(self, node: Flatten, **kwargs: Any) -> None:
        inputs = self.tf_ops[node.input_ops['input'].name]

        with self._tf_graph.as_default():
            output = tf.layers.flatten(inputs, name=node.name)
        self.tf_ops[node.name] = output

    def run_forward_dropout(self, node: Dropout, **kwargs: Any) -> None:
        x = self.tf_ops[node.input_ops['data'].name]
        keep_prob = 1 - node.ratio

        with self._tf_graph.as_default():
            output = tf.nn.dropout(x, keep_prob, name=node.name)
        self.tf_ops[node.name] = output

    def run_forward_gemm(self, node: Gemm, **kwargs: Any) -> None:
        raise NotImplementedError(f'conversion of {node.op_type} is not supported yet.')
