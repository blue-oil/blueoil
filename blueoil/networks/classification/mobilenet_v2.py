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

from blueoil.nn.networks.classification.base import Base


class MobileNetV2(Base):
    """MobileNet v2
    """
    version = 1.0

    def __init__(
            self,
            # the params value following:
            # https://github.com/tensorflow/models/blob/505f554c6417931c96b59516f14d1ad65df6dbc5/research/slim/nets/mobilenet/mobilenet.py#L417 # NOQA
            dropout_keep_prob=0.8,
            batch_norm_decay=0.997,
            stddev=0.09,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu6
        self.custom_getter = None

        self.channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'

        self.stddev = stddev
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_norm_decay = batch_norm_decay

        # Original model use truncated_normal as kernel init, but xavier is better than it in Cifar10 experiment.
        self.conv2d = functools.partial(tf.layers.conv2d,
                                        data_format=self.channels_data_format,
                                        padding="SAME",
                                        # kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        use_bias=False,)

        self.batch_norm = functools.partial(tf.contrib.layers.batch_norm,
                                            decay=self.batch_norm_decay,
                                            scale=True,
                                            center=True,
                                            updates_collections=None,
                                            data_format=self.data_format)

    def _inverted_bottleneck(self, x, up_sample_rate, channels, subsample):
        input_x = x
        input_shape = x.get_shape()

        if self.data_format == 'NHWC':
            input_channel = input_shape.as_list()[-1]
        else:
            input_channel = input_shape.as_list()[1]

        expanded_filters_size = up_sample_rate * input_channel

        with tf.compat.v1.variable_scope('inverted_bottleneck{}_{}_{}'.format(self.i, up_sample_rate, subsample)):
            self.i += 1
            if expanded_filters_size > input_channel:
                x = self.conv2d(x, expanded_filters_size, 1, activation=None)
                x = self.batch_norm(x, is_training=self.is_training)
                x = self.activation(x)

            if subsample:
                stride = 2
            else:
                stride = 1
            x = tf.contrib.layers.separable_conv2d(x, None, 3, 1, stride=stride,
                                                   activation_fn=None,
                                                   data_format=self.data_format,
                                                   normalizer_fn=functools.partial(
                                                       self.batch_norm,
                                                       is_training=self.is_training
                                                   ))
            x = self.activation(x)

            x = self.conv2d(x, channels, 1, activation=None)
            x = self.batch_norm(x, is_training=self.is_training)

            if input_channel == channels:
                x = tf.add(input_x, x)
            return x

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        self.is_training = is_training
        self.images = images

        self.i = 0
        with tf.compat.v1.variable_scope('init_conv'):
            # For cifar10, I use strides `1` instead of `2`.
            x = self.conv2d(self.images, filters=32, kernel_size=3, strides=1)
            x = self.batch_norm(x, is_training=self.is_training)
            x = self.activation(x)

        x = self._inverted_bottleneck(x, 1, 16, 0)
        x = self._inverted_bottleneck(x, 6, 24, 1)
        x = self._inverted_bottleneck(x, 6, 24, 0)
        x = self._inverted_bottleneck(x, 6, 32, 1)
        x = self._inverted_bottleneck(x, 6, 32, 0)
        x = self._inverted_bottleneck(x, 6, 32, 0)
        x = self._inverted_bottleneck(x, 6, 64, 1)
        x = self._inverted_bottleneck(x, 6, 64, 0)
        x = self._inverted_bottleneck(x, 6, 64, 0)
        x = self._inverted_bottleneck(x, 6, 64, 0)
        x = self._inverted_bottleneck(x, 6, 96, 0)
        x = self._inverted_bottleneck(x, 6, 96, 0)
        x = self._inverted_bottleneck(x, 6, 96, 0)
        x = self._inverted_bottleneck(x, 6, 160, 1)
        x = self._inverted_bottleneck(x, 6, 160, 0)
        x = self._inverted_bottleneck(x, 6, 160, 0)
        x = self._inverted_bottleneck(x, 6, 320, 0)

        self.i += 1
        with tf.compat.v1.variable_scope('conv' + str(self.i)):
            x = self.conv2d(x, 1280, 1)
            x = self.batch_norm(x, is_training=self.is_training)
            x = self.activation(x)

        h = x.get_shape()[1].value if self.data_format == 'NHWC' else x.get_shape()[2].value
        w = x.get_shape()[2].value if self.data_format == 'NHWC' else x.get_shape()[3].value
        x = tf.layers.average_pooling2d(name='pool7',
                                        inputs=x,
                                        pool_size=[h, w],
                                        padding='VALID',
                                        strides=1,
                                        data_format=self.channels_data_format)

        x = tf.contrib.layers.dropout(x, keep_prob=self.dropout_keep_prob, is_training=is_training)

        self.i += 1
        with tf.compat.v1.variable_scope('conv' + str(self.i)):
            x = self.conv2d(x, filters=self.num_classes, kernel_size=1, use_bias=True)

        # disable heatmap for faster learning.
        # self._heatmap_layer = x
        self.base_output = tf.reshape(x, [-1, self.num_classes], name='last_layer_reshape')

        return self.base_output
