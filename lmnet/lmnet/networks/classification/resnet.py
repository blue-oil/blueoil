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
from functools import partial
import tensorflow as tf

from lmnet.networks.classification.base import Base
from lmnet.layers import average_pooling2d, batch_norm, conv2d, fully_connected


class Resnet(Base):
    version = ""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # num_residual: all layer number is 2 + (num_residual * 4 * 2).
        self.num_residual = 2
        self.activation = tf.nn.relu
        self.before_last_activation = tf.nn.relu
        self.is_debug = False

    def res_block(self, inputs, in_filters, out_filters, strides,
                  is_training, activation):
        with tf.variable_scope('sub1'):
            conv1 = conv2d(
                "conv1",
                inputs,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=False,
                strides=strides,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                is_debug=self.is_debug,
            )
            bn1 = batch_norm("bn1", conv1, is_training=is_training, epsilon=1e-5,)
            relu1 = activation(bn1)

        with tf.variable_scope('sub2'):
            conv2 = conv2d(
                "conv2",
                relu1,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=False,
                strides=1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                is_debug=self.is_debug,
            )

            bn2 = batch_norm("bn2", conv2, is_training=is_training, epsilon=1e-5,)

        with tf.variable_scope('sub_add'):
            if in_filters != out_filters:
                inputs = tf.nn.avg_pool(
                    inputs,
                    ksize=[1, strides, strides, 1],
                    strides=[1, strides, strides, 1],
                    padding='VALID'
                )
                inputs = tf.pad(
                    inputs,
                    [[0, 0], [0, 0], [0, 0], [(out_filters - in_filters)//2, (out_filters - in_filters)//2]]
                )

        added = bn2 + inputs

        output = activation(added)

        return output

    def base(self, images, is_training):
        self.images = self.input = images

        with tf.variable_scope("init"):
            self.conv1 = conv2d(
                "conv1",
                self.images,
                filters=64,
                kernel_size=7,
                strides=2,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                is_debug=self.is_debug,
            )

            self.bn1 = batch_norm("bn1", self.conv1, is_training=is_training, epsilon=1e-5,)
            self.relu1 = self.activation(self.bn1)

        self.pool1 = tf.layers.max_pooling2d(self.relu1, pool_size=3, strides=2, padding='SAME',)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit1_{}".format(i)):
                if i == 0:
                    out = self.res_block(self.pool1, in_filters=64, out_filters=64, strides=1,
                                         is_training=is_training, activation=self.activation)
                else:
                    out = self.res_block(out, in_filters=64, out_filters=64, strides=1,
                                         is_training=is_training, activation=self.activation)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit2_{}".format(i)):
                if i == 0:
                    out = self.res_block(out, in_filters=64, out_filters=128, strides=2,
                                         is_training=is_training, activation=self.activation)
                else:
                    out = self.res_block(out, in_filters=128, out_filters=128, strides=1,
                                         is_training=is_training, activation=self.activation)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit3_{}".format(i)):
                if i == 0:
                    out = self.res_block(out, in_filters=128, out_filters=256, strides=2,
                                         is_training=is_training, activation=self.activation)
                else:
                    out = self.res_block(out, in_filters=256, out_filters=256, strides=1,
                                         is_training=is_training, activation=self.activation)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit4_{}".format(i)):
                if i == 0:
                    out = self.res_block(out, in_filters=256, out_filters=512, strides=2,
                                         is_training=is_training, activation=self.activation)
                elif i == (self.num_residual - 1):
                    out = self.res_block(out, in_filters=512, out_filters=512, strides=1,
                                         is_training=is_training, activation=self.before_last_activation)
                else:
                    out = self.res_block(out, in_filters=512, out_filters=512, strides=1,
                                         is_training=is_training, activation=self.activation)

        # global average pooling
        h = out.get_shape()[1].value
        w = out.get_shape()[2].value
        self.global_average_pool = average_pooling2d(
            "global_average_pool", out, pool_size=[h, w], padding="VALID", is_debug=self.is_debug,)

        self._heatmap_layer = None
        self.fc = fully_connected(
            "fc", self.global_average_pool, filters=self.num_classes, activation=None,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        )

        return self.fc


class ResnetCifar(Base):
    version = ""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # num_residual: all layer number is 2 + (num_residual * 3 * 2).
        self.num_residual = 3
        self.activation = tf.nn.relu
        self.before_last_activation = tf.nn.relu
        self.is_debug = True

    def res_block(self, inputs, in_filters, out_filters, strides,
                  is_training, activation):
        with tf.variable_scope('sub1'):
            conv1 = conv2d(
                "conv1",
                inputs,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=False,
                strides=strides,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                is_debug=self.is_debug,
            )
            bn1 = batch_norm("bn1", conv1, is_training=is_training, epsilon=1e-5,)
            relu1 = activation(bn1)

        with tf.variable_scope('sub2'):
            conv2 = conv2d(
                "conv2",
                relu1,
                filters=out_filters,
                kernel_size=3,
                activation=None,
                use_bias=False,
                strides=1,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                is_debug=self.is_debug,
            )

            bn2 = batch_norm("bn2", conv2, is_training=is_training, epsilon=1e-5,)

        with tf.variable_scope('sub_add'):
            if in_filters != out_filters:
                inputs = tf.nn.avg_pool(
                    inputs,
                    ksize=[1, strides, strides, 1],
                    strides=[1, strides, strides, 1],
                    padding='VALID'
                )
                inputs = tf.pad(
                    inputs,
                    [[0, 0], [0, 0], [0, 0], [(out_filters - in_filters)//2, (out_filters - in_filters)//2]]
                )

        added = bn2 + inputs

        output = activation(added)

        return output

    def base(self, images, is_training):
        self.images = self.input = images

        with tf.variable_scope("init"):
            self.conv1 = conv2d(
                "conv1",
                self.images,
                filters=16,
                kernel_size=3,
                activation=None,
                use_bias=False,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                is_debug=self.is_debug,
            )

            self.bn1 = batch_norm("bn1", self.conv1, is_training=is_training, epsilon=1e-5,)
            self.relu1 = self.activation(self.bn1)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit1_{}".format(i)):
                if i == 0:
                    out = self.res_block(self.relu1, in_filters=16, out_filters=16, strides=1,
                                         is_training=is_training, activation=self.activation)
                else:
                    out = self.res_block(out, in_filters=16, out_filters=16, strides=1,
                                         is_training=is_training, activation=self.activation)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit2_{}".format(i)):
                if i == 0:
                    out = self.res_block(out, in_filters=16, out_filters=32, strides=2,
                                         is_training=is_training, activation=self.activation)
                else:
                    out = self.res_block(out, in_filters=32, out_filters=32, strides=1,
                                         is_training=is_training, activation=self.activation)

        for i in range(0, self.num_residual):
            with tf.variable_scope("unit3_{}".format(i)):
                if i == 0:
                    out = self.res_block(out, in_filters=32, out_filters=64, strides=2,
                                         is_training=is_training, activation=self.activation)
                elif i == (self.num_residual - 1):
                    out = self.res_block(out, in_filters=64, out_filters=64, strides=1,
                                         is_training=is_training, activation=self.before_last_activation)
                else:
                    out = self.res_block(out, in_filters=64, out_filters=64, strides=1,
                                         is_training=is_training, activation=self.activation)

        # global average pooling
        h = out.get_shape()[1].value
        w = out.get_shape()[2].value
        self.global_average_pool = average_pooling2d(
            "global_average_pool", out, pool_size=[h, w], padding="VALID", is_debug=self.is_debug,)

        self._heatmap_layer = None
        self.fc = fully_connected(
            "fc", self.global_average_pool, filters=self.num_classes, activation=None,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        )

        return self.fc


class ResnetCifarQuantize(ResnetCifar):

    """Quantize Network."""

    def __init__(
            self,
            quantize_first_convolution=True,
            quantize_last_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        """
        Args:
            quantize_first_convolution(bool): use quantization in first conv.
            quantize_last_convolution(bool): use quantization in last conv.
            weight_quantizer (callable): weight quantizer.
            weight_quantize_kwargs(dict): Initialize kwargs for weight quantizer.
            activation_quantizer (callable): activation quantizer
            activation_quantize_kwargs(dict): Initialize kwargs for activation quantizer.
        """

        super().__init__(
            *args,
            **kwargs,
        )

        self.quantize_first_convolution = quantize_first_convolution
        self.quantize_last_convolution = quantize_last_convolution

        activation_quantizer_kwargs = activation_quantizer_kwargs if not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if not None else {}

        if weight_quantizer:
            assert callable(weight_quantizer)
            self.weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        else:
            self.weight_quantization = None

        assert callable(activation_quantizer)
        self.activation = activation_quantizer(**activation_quantizer_kwargs)

        if self.quantize_last_convolution:
            self.before_last_activation = self.activation

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
        with tf.variable_scope(name):
            if "kernel" == var.op.name.split("/")[-1]:

                if not quantize_first_convolution:
                    if var.op.name.startswith("block_1/"):
                        return var

                if not quantize_last_convolution:
                    if var.op.name.startswith("conv_23/"):
                        return var

                # Apply weight quantize to variable whose last word of name is "kernel".
                quantized_kernel = weight_quantization(var)
                tf.summary.histogram("quantized_kernel", quantized_kernel)
                return quantized_kernel

        return var

    def base(self, images, is_training):
        if self.weight_quantization:
            custom_getter = partial(
                self._quantized_variable_getter,
                self.weight_quantization,
                self.quantize_first_convolution,
                self.quantize_last_convolution,
            )
        else:
            custom_getter = None
        with tf.variable_scope("", custom_getter=custom_getter):
            return super().base(images, is_training)
