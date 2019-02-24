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
import tensorflow as tf
from lmnet.layers import conv2d, batch_norm


def darknet(name, inputs, filters, kernel_size, is_training=tf.constant(False), activation=None, data_format="NHWC"):
    """Darknet19 block.

    Do convolution, batch_norm, bias, leaky_relu activation.
    Ref: https://arxiv.org/pdf/1612.08242.pdf
         https://github.com/pjreddie/darknet/blob/3bf2f342c03b0ad22efd799d5be9990c9d792354/cfg/darknet19.cfg
         https://github.com/pjreddie/darknet/blob/8215a8864d4ad07e058acafd75b2c6ff6600b9e8/cfg/yolo.2.0.cfg
    """
    if data_format == "NCHW":
        channel_data_format = "channels_first"
    elif data_format == "NHWC":
        channel_data_format = "channels_last"
    else:
        raise ValueError("data format must be 'NCHW' or 'NHWC'. got {}.".format(data_format))

    with tf.variable_scope(name):
        if activation is None:
            def activation(x): return tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

        conv = conv2d("conv", inputs, filters=filters, kernel_size=kernel_size,
                      activation=None, use_bias=False, data_format=channel_data_format,
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),)  # he initializer

        # TODO(wakisaka): Should be the same as darknet batrch norm.
        # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/layers/python/layers/layers.py
        # https://github.com/pjreddie/darknet/blob/8215a8864d4ad07e058acafd75b2c6ff6600b9e8/src/batchnorm_layer.c#L135
        batch_normed = batch_norm("bn", conv, is_training=is_training, decay=0.99, scale=True, center=True,
                                  data_format=data_format)
        tf.summary.histogram("batch_normed", batch_normed)

        output = activation(batch_normed)
        tf.summary.histogram("output", output)

        return output


def lmnet_block(
        name,
        inputs,
        filters,
        kernel_size,
        custom_getter=None,
        is_training=tf.constant(True),
        activation=None,
        use_bias=True,
        use_batch_norm=True,
        is_debug=False,
        data_format='channels_last',
        batch_norm_decay=0.99,
):
    """Block used in lmnet

    Combine convolution, bias, weights quantization and activation quantization as one layer block.

    Args:
        name(str): Block name, as scope name.
        inputs(tf.Tensor): Inputs.
        filters(int): Number of filters for convolution.
        kernel_size(int): Kernel size.
        custom_getter(callable): Custom getter for `tf.variable_scope`.
        is_training(tf.constant): Flag if training or not.
        activation(callable): Activation function.
        use_bias(bool): If use bias.
        use_batch_norm(bool): If use batch norm.
        is_debug(bool): If is debug.
        data_format(string): channels_last for NHWC. channels_first for NCHW. Default is channels_last.
    Returns:
        tf.Tensor: Output of current layer block.
    """
    with tf.variable_scope(name, custom_getter=custom_getter):
        conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, padding='SAME', use_bias=False,
                                data_format=data_format)

        if use_batch_norm:
            # TODO(wenhao) hw supports `tf.contrib.layers.batch_norm` currently. change it when supported.
            # batch_normed = tf.layers.batch_normalization(conv,
            #                                              momentum=0.99,
            #                                              scale=True,
            #                                              center=True,
            #                                              training=is_training)
            four_letter_data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
            batch_normed = tf.contrib.layers.batch_norm(conv,
                                                        decay=batch_norm_decay,
                                                        scale=True,
                                                        center=True,
                                                        updates_collections=None,
                                                        is_training=is_training,
                                                        data_format=four_letter_data_format)

        else:
            batch_normed = conv

        if use_bias:
            bias = tf.get_variable('bias', shape=filters, initializer=tf.zeros_initializer)
            biased = batch_normed + bias
        else:
            biased = batch_normed

        if activation:
            output = activation(biased)
        else:
            output = biased

        if is_debug:
            tf.summary.histogram('conv', conv)
            tf.summary.histogram('batch_normed', batch_normed)
            tf.summary.histogram('biased', biased)
            tf.summary.histogram('output', output)

        return output
