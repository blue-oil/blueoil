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
from tensorflow.python.framework import ops

from lmnet.blocks import lmnet_block
from lmnet.layers.experiment import max_unpool_with_argmax
from lmnet.networks.segmentation.base import SegnetBase


class LmSegnetV0(SegnetBase):
    """LM customized SegNet Network."""

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

    def _max_pool_with_argmax(self, inputs=None, ksize=None, strides=None, padding=None, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])
        output, argmax = tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding, name=name)
        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output, argmax

    def _unpool_with_argmax(self, inputs=None, mask=None, ksize=None, name=''):
        with ops.name_scope(name, "Unpool"):
            return max_unpool_with_argmax(inputs,
                                          mask,
                                          ksize,
                                          self.data_format)

    def base(self, images, is_training, *args, **kwargs):
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        max_pool_with_argmax = functools.partial(self._max_pool_with_argmax,
                                                 ksize=(1, 2, 2, 1),
                                                 strides=(1, 2, 2, 1),
                                                 padding='SAME')

        unpool_with_argmax = functools.partial(self._unpool_with_argmax,
                                               ksize=(1, 2, 2, 1))
        self.images = images

        x = lmnet_block('conv1', images, 32, 3)
        x, i_1 = max_pool_with_argmax(name='pool1', inputs=x)
        x = lmnet_block('conv2', x, 64, 3)
        x, i_2 = max_pool_with_argmax(name='pool2', inputs=x)
        x = lmnet_block('conv3', x, 128, 3)
        x, i_3 = max_pool_with_argmax(name='pool3', inputs=x)
        x = lmnet_block('conv4', x, 256, 3)
        x, i_4 = max_pool_with_argmax(name='pool4', inputs=x)
        x = lmnet_block('conv5', x, 256, 3)
        x, i_5 = max_pool_with_argmax(name='pool5', inputs=x)

        x = unpool_with_argmax(name='unpool6', inputs=x, mask=i_5)
        x = lmnet_block('conv6', x, 256, 3)
        x = unpool_with_argmax(name='unpool7', inputs=x, mask=i_4)
        x = lmnet_block('conv7', x, 128, 3)
        x = unpool_with_argmax(name='unpool8', inputs=x, mask=i_3)
        x = lmnet_block('conv8', x, 64, 3)
        x = unpool_with_argmax(name='unpool9', inputs=x, mask=i_2)
        x = lmnet_block('conv9', x, 32, 3)
        x = unpool_with_argmax(name='unpool10', inputs=x, mask=i_1)
        x = lmnet_block('conv10', x, 32, 3)
        x = lmnet_block('conv11', x, self.num_classes, 3)

        return x


class LmSegnetV0Quantize(LmSegnetV0):
    """LM customized Segnet quantize network.

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
    """

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
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
