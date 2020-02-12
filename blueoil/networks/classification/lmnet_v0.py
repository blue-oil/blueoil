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

from blueoil.blocks import lmnet_block
from blueoil.networks.classification.base import Base


class LmnetV0(Base):
    """Lmnet network for classification, version 0.
    """
    version = 0.01

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(lmnet_block,
                                 activation=self.activation,
                                 custom_getter=self.custom_getter,
                                 is_training=is_training,
                                 is_debug=self.is_debug,
                                 use_bias=False,
                                 data_format=channels_data_format)

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)
        _max_pooling2d = functools.partial(tf.layers.max_pooling2d, pool_size=2, strides=2, padding='SAME',
                                           data_format=channels_data_format)

        self.images = images

        x = _lmnet_block('conv1', images, 32, 3)
        x = _max_pooling2d(name='pool1', inputs=x)
        x = _lmnet_block('conv2', x, 64, 3)
        x = _max_pooling2d(name='pool2', inputs=x)
        x = _lmnet_block('conv3', x, 128, 3)
        x = _max_pooling2d(name='pool3', inputs=x)
        x = _lmnet_block('conv4', x, 256, 3)
        x = _max_pooling2d(name='pool4', inputs=x)
        x = _lmnet_block('conv5', x, 256, 3)
        x = _max_pooling2d(name='pool5', inputs=x)
        x = _lmnet_block('conv6', x, 64, 1, activation=tf.nn.relu)

        x = tf.layers.dropout(x, training=is_training)

        kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        x = tf.layers.conv2d(name='conv7',
                             inputs=x,
                             filters=self.num_classes,
                             kernel_size=1,
                             kernel_initializer=kernel_initializer,
                             activation=None,
                             use_bias=True,
                             data_format=channels_data_format)

        self._heatmap_layer = x

        h = x.get_shape()[1].value if self.data_format == 'NHWC' else x.get_shape()[2].value
        w = x.get_shape()[2].value if self.data_format == 'NHWC' else x.get_shape()[3].value
        x = tf.layers.average_pooling2d(name='pool7',
                                        inputs=x,
                                        pool_size=[h, w],
                                        padding='VALID',
                                        strides=1,
                                        data_format=channels_data_format)

        self.base_output = tf.reshape(x, [-1, self.num_classes], name='pool7_reshape')

        return self.base_output


class LmnetV0Quantize(LmnetV0):
    """Lmnet quantize network for classification, version 1.0

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `blueoil.nn.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `blueoil.nn.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
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

        assert weight_quantizer
        assert activation_quantizer

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
        with tf.compat.v1.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
