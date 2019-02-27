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

from lmnet.networks.classification.base import Base


def conv(inputs, filters, kernel_size):
    out = tf.layers.conv2d(
        inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding='SAME',
        use_bias=False)
    return out


def batch_norm(inputs, is_training):
    out = tf.contrib.layers.batch_norm(
        inputs,
        scale=True,
        center=True,
        updates_collections=None,
        is_training=is_training)

    return out


class DoubleConcatQuantize(Base):
    """
    """
    version = 1.0

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
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
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var

    def concat(self, name, inputs, channel, kernel):
        pass

    def base(self, images, is_training, *args, **kwargs):
        self.images = images

        with tf.variable_scope("first"):
            x = conv(images, 32, 1)
            x = batch_norm(x, is_training)
            x = self.activation(x)
            stock = x

        with tf.variable_scope("", custom_getter=self.custom_getter):
            with tf.variable_scope("second"):
                x = conv(x, 32, 3)
                x = batch_norm(x, is_training)
                x = self.activation(x)

                x = tf.concat([stock, x], axis=3)

                stock = x

            with tf.variable_scope("third"):
                x = conv(x, 32, 3)
                x = batch_norm(x, is_training)
                x = self.activation(x)

                x = tf.concat([stock, x], axis=3)

        with tf.variable_scope("fourth"):
            x = conv(x, self.num_classes, 3)
            x = batch_norm(x, is_training)

            h = x.get_shape()[1].value if self.data_format == 'NHWC' else x.get_shape()[2].value
            w = x.get_shape()[2].value if self.data_format == 'NHWC' else x.get_shape()[3].value
            x = tf.layers.average_pooling2d(name='pool7',
                                            inputs=x,
                                            pool_size=[h, w],
                                            padding='VALID',
                                            strides=1)

            self.base_output = tf.reshape(x, [-1, self.num_classes], name='pool7_reshape')

        return self.base_output
