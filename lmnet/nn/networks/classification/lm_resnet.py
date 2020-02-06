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

from nn.layers import fully_connected
from nn.networks.classification.base import Base


class LmResnet(Base):
    """Residual network (ResNet) of 18-layers for classification

    This ResNet-18 is modified from the ImageNet version of ResNet-18 of the original paper
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

    - The first layer is 3x3 convolution layer with stride 1 instead of 7x7 conv with stride 2,
      like the CIFAR-10 version of ResNet-18 in the paper.
    - The 3x3 max pooling with stride 2 is not used in this architecture.
    - In each residual block, batch normalization (BN) is after the add, to be specific, a
      pre-activation variant of residual block is used.
    - Utilizing Space-to-Depth operator for each the transition layer, convolution op with
      strides of 2 is replaced with space-to-depth.
    - Note currently this ResNet-18 only supports NHWC data format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.data_format == 'NHWC'
        self.custom_getter = None
        self.activation = tf.nn.relu
        self.init_ch = 64
        self.num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
        }[18]

    @staticmethod
    def _batch_norm(inputs, training):
        return tf.contrib.layers.batch_norm(
            inputs,
            decay=0.997,
            updates_collections=None,
            is_training=training,
            activation_fn=None,
            center=True,
            scale=True)

    @staticmethod
    def _conv2d_fix_padding(inputs, filters, kernel_size, strides):
        """Convolution layer deals with stride of 2"""
        if strides == 2:
            inputs = tf.space_to_depth(inputs, block_size=2, name="pool")

        return tf.layers.conv2d(
            inputs, filters, kernel_size,
            padding="SAME",
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            use_bias=False)

    def basicblock(self, x, out_ch, strides, training):
        """Basic building block of single residual function"""
        in_ch = x.get_shape().as_list()[1 if self.data_format in {'NCHW', 'channels_first'} else 3]
        shortcut = x

        x = self._batch_norm(x, training)
        x = self.activation(x)

        x = self._conv2d_fix_padding(x, out_ch, 3, strides)
        x = self._batch_norm(x, training)
        x = self.activation(x)

        x = self._conv2d_fix_padding(x, out_ch, 3, 1)

        if strides == 2:
            shortcut = tf.nn.avg_pool(shortcut, ksize=[1, strides, strides, 1],
                                      strides=[1, strides, strides, 1], padding='VALID')
        if in_ch != out_ch:
            shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0],
                              [(out_ch - in_ch) // 2, (out_ch - in_ch + 1) // 2]])
        return shortcut + x

    def resnet_group(self, x, out_ch, count, strides, training, name):
        with tf.compat.v1.variable_scope(name, custom_getter=self.custom_getter):
            for i in range(0, count):
                with tf.compat.v1.variable_scope('block{}'.format(i)):
                    x = self.basicblock(x, out_ch,
                                        strides if i == 0 else 1,
                                        training)
        return x

    def base(self, images, is_training):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if it is training or not.
        Returns:
            tf.Tensor: Inference result.
        """
        self.images = images

        x = self._conv2d_fix_padding(images, self.init_ch, 3, 1)
        x = self.resnet_group(x, self.init_ch * 1, self.num_blocks[0], 1, is_training, 'group0')
        x = self.resnet_group(x, self.init_ch * 2, self.num_blocks[1], 2, is_training, 'group1')
        x = self.resnet_group(x, self.init_ch * 4, self.num_blocks[2], 2, is_training, 'group2')
        x = self.resnet_group(x, self.init_ch * 8, self.num_blocks[3], 2, is_training, 'group3')
        x = self._batch_norm(x, is_training)
        x = tf.nn.relu(x)

        # global average pooling
        h = x.get_shape()[1].value
        w = x.get_shape()[2].value
        x = tf.layers.average_pooling2d(name="gap", inputs=x, pool_size=[h, w], padding="VALID", strides=1)

        output = fully_connected("linear", x, filters=self.num_classes, activation=None)

        return output


class LmResnetQuantize(LmResnet):
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
