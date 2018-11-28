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
import functools

import tensorflow as tf

from lmnet.blocks import lmnet_block
from lmnet.networks.classification.base import Base
from lmnet.networks.base_quantize import BaseQuantize

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg7Network(Base):
    def __init__(
            self,
            quantize_first_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            weight_decay_rate=None,
            classes=(),
            optimizer_class=tf.train.GradientDescentOptimizer,
            optimizer_kwargs=None,
            is_debug=False,
            learning_rate_func=None,
            learning_rate_kwargs=None,
            image_size=(),  # [height, width]
            batch_size=64,
            data_format='NHWC',
            *args,
            **kwargs,
    ):

        self.num_classes = len(classes)
        self.data_format = data_format
        self.custom_getter = None
        self.activation = tf.nn.relu

        super().__init__(
            weight_decay_rate,
            is_debug,
            optimizer_class,
            optimizer_kwargs,
            learning_rate_func,
            learning_rate_kwargs,
            classes,
            image_size,
            batch_size,
            data_format,
        )

    def base(self, images, is_training=None):
        self.images = images

        self.input = self.convert_rbg_to_bgr(images)
        self.input = tf.cast(images, dtype=tf.float32)

        self.conv1 = self.conv_layer("conv1", self.input, filters=128, kernel_size=3)
        self.conv2 = self.conv_layer("conv2", self.conv1, filters=128, kernel_size=3)

        self.pool1 = self.max_pool("pool1", self.conv2, kernel_size=3, strides=2)

        self.conv3 = self.conv_layer("conv3", self.pool1, filters=256, kernel_size=3)
        self.conv4 = self.conv_layer("conv4", self.conv3, filters=256, kernel_size=3)

        self.pool2 = self.max_pool("pool2", self.conv4, kernel_size=3, strides=2)

        self.conv5 = self.conv_layer("conv5", self.pool2, filters=512, kernel_size=3)
        self.conv6 = self.conv_layer("conv6", self.conv5, filters=512, kernel_size=3)
        
        self.pool3 = self.max_pool("pool3", self.conv6, kernel_size=3, strides=2)

        self.flatten = tf.contrib.layers.flatten(self.pool3)

        self.fc1 = self.fc_layer("fc1", self.flatten, filters=self.num_classes, activation=None)
        self.base_output = tf.reshape(self.fc1, [-1, self.num_classes], name='fc1_reshape')

        return self.base_output

    def conv_layer(
        self,
        name,
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding="SAME",
        activation=tf.nn.relu,
        *args,
        **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()
        
        if self.data_format == 'NHWC':
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'

        with tf.variable_scope(name, custom_getter=self.custom_getter):
            output = tf.layers.conv2d(name=name,
                                      inputs=inputs,
                                      filters=filters,
                                      kernel_size=kernel_size,
                                      kernel_initializer=kernel_initializer,
                                      activation=tf.nn.relu,
                                      #use_bias=False,
                                      bias_initializer=biases_initializer,
                                      padding=padding,
                                      strides=strides,
                                      data_format=data_format)

        return output

    def fc_layer(
            self,
            name,
            inputs,
            filters,
            *args,
            **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()

        output = tf.contrib.layers.fully_connected(
            inputs = inputs,
            num_outputs=filters,
            weights_initializer=kernel_initializer,
            biases_initializer=biases_initializer,
        )

        return output

    def convert_rbg_to_bgr(self, rgb_images):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_images)

        bgr_images = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        return bgr_images

    def max_pool(
            self,
            name,
            inputs,
            kernel_size, 
            strides
    ):
        if self.data_format == 'NHWC':
            data_format = 'channels_last'
        else:
            data_format = 'channels_first'

        output = tf.layers.max_pooling2d(
            name=name,
            inputs=inputs,
            pool_size=kernel_size,
            strides=strides,
            data_format=data_format,
        )

        return output


class Vgg7Quantize(Vgg7Network, BaseQuantize):
    version = 1.0

    def __init__(
            self,        
            quantize_first_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        Vgg7Network.__init__(
            self,
            *args,
            **kwargs
        )

        BaseQuantize.__init__(
            self,
            activation_quantizer,
            activation_quantizer_kwargs,
            weight_quantizer,
            weight_quantizer_kwargs,
            quantize_first_convolution,
        )
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               quantize_first_convolution=self.quantize_first_convolution,
                                               weight_quantization=self.weight_quantization)

        
    @staticmethod
    def _quantized_variable_getter(getter, name, quantize_first_convolution, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """

        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if not quantize_first_convolution:
                if var.op.name.startswith("conv1/"):
                    return var

            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
