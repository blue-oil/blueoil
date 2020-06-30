# -*- coding: utf-8 -*-
# Copyright 2020 The Blueoil Authors. All Rights Reserved.
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
from functools import partial

import tensorflow as tf

from blueoil.networks.classification.base import Base
from blueoil.layers import batch_norm, conv2d


class SampleNetwork(Base):
    """Sample network with simple layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

    def base(self, images, is_training):
        assert self.data_format == "NHWC"
        channel_data_format = "channels_last"

        self.inputs = self.images = images

        with tf.compat.v1.variable_scope("block_1"):
            conv = conv2d("conv", self.inputs, filters=32, kernel_size=3,
                          activation=None, use_bias=False, data_format=channel_data_format,
                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0))
            batch_normed = batch_norm("bn", conv, is_training=is_training, decay=0.99, scale=True, center=True,
                                      data_format=self.data_format)
            self.block_1 = self.activation(batch_normed)

        self.block_last = conv2d("block_last", self.block_1, filters=self.num_classes, kernel_size=1,
                                 activation=None, use_bias=True, is_debug=self.is_debug,
                                 kernel_initializer=tf.compat.v1.random_normal_initializer(mean=0.0, stddev=0.01),
                                 data_format=channel_data_format)

        h = self.block_last.get_shape()[1].value
        w = self.block_last.get_shape()[2].value
        self.pool = tf.compat.v1.layers.average_pooling2d(name='global_average_pool', inputs=self.block_last,
                                                          pool_size=[h, w], padding='VALID', strides=1,
                                                          data_format=channel_data_format)
        self.base_output = tf.reshape(self.pool, [-1, self.num_classes], name="pool_reshape")

        return self.base_output


class SampleNetworkQuantize(SampleNetwork):
    """Quantize Sample Network."""

    def __init__(
            self,
            quantize_first_convolution=True,
            quantize_last_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs={},
            weight_quantizer=None,
            weight_quantizer_kwargs={},
            *args,
            **kwargs
    ):
        """
        Args:
            quantize_first_convolution(bool): use quantization in first conv.
            quantize_last_convolution(bool): use quantization in last conv.
            weight_quantizer (callable): weight quantizer.
            weight_quantizer_kwargs(dict): Initialize kwargs for weight quantizer.
            activation_quantizer (callable): activation quantizer
            activation_quantizer_kwargs(dict): Initialize kwargs for activation quantizer.
        """

        super().__init__(*args, **kwargs)

        self.quantize_first_convolution = quantize_first_convolution
        self.quantize_last_convolution = quantize_last_convolution

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        self.weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.activation = activation_quantizer(**activation_quantizer_kwargs)

    @staticmethod
    def _quantized_variable_getter(
            weight_quantization,
            quantize_first_convolution,
            quantize_last_convolution,
            getter,
            name,
            *args,
            **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            weight_quantization: Callable object which quantize variable.
            quantize_first_convolution(bool): Use quantization in first conv.
            quantize_last_convolution(bool): Use quantization in last conv.
            getter: Default from tensorflow.
            name: Default from tensorflow.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.compat.v1.variable_scope(name):
            if "kernel" == var.op.name.split("/")[-1]:

                if not quantize_first_convolution:
                    if var.op.name.startswith("block_1/"):
                        return var

                if not quantize_last_convolution:
                    if var.op.name.startswith("block_last/"):
                        return var

                # Apply weight quantize to variable whose last word of name is "kernel".
                quantized_kernel = weight_quantization(var)
                tf.compat.v1.summary.histogram("quantized_kernel", quantized_kernel)
                return quantized_kernel

        return var

    def base(self, images, is_training):
        custom_getter = partial(
            self._quantized_variable_getter,
            self.weight_quantization,
            self.quantize_first_convolution,
            self.quantize_last_convolution,
        )
        with tf.compat.v1.variable_scope("", custom_getter=custom_getter):
            return super().base(images, is_training)
