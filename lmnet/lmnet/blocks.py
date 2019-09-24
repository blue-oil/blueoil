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

from lmnet.layers import batch_norm, conv2d


# TODO(wakisaka): should be replace to conv_bn_act().
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

        # TODO(wakisaka): Should be the same as darknet batch norm.
        # https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/contrib/layers/python/layers/layers.py
        # https://github.com/pjreddie/darknet/blob/8215a8864d4ad07e058acafd75b2c6ff6600b9e8/src/batchnorm_layer.c#L135
        batch_normed = batch_norm("bn", conv, is_training=is_training, decay=0.99, scale=True, center=True,
                                  data_format=data_format)
        tf.summary.histogram("batch_normed", batch_normed)

        output = activation(batch_normed)
        tf.summary.histogram("output", output)

        return output


# TODO(wakisaka): should be replace to conv_bn_act().
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
                                                        decay=0.99,
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


def conv_bn_act(
        name,
        inputs,
        filters,
        kernel_size,
        weight_decay_rate=0.0,
        is_training=tf.constant(False),
        activation=None,
        batch_norm_decay=0.99,
        data_format="NHWC",
        enable_detail_summary=False,
):
    """Block of convolution -> batch norm -> activation.

    Args:
        name (str): Block name, as scope name.
        inputs (tf.Tensor): Inputs.
        filters (int): Number of filters (output channel) for convolution.
        kernel_size (int): Kernel size.
        weight_decay_rate (float): Number of L2 regularization be applied to convolution weights.
           Need `tf.losses.get_regularization_loss()` in loss function to apply this parameter to loss.
        is_training (tf.constant): Flag if training or not for batch norm.
        activation (callable): Activation function.
        batch_norm_decay (float): Batch norm decay rate.
        data_format (string):  Format for inputs data. NHWC or NCHW.
        enable_detail_summary (bool): Flag for summarize feature maps for each operation on tensorboard.
    Returns:
        output (tf.Tensor): Output of this block.

    """
    if data_format == "NCHW":
        channel_data_format = "channels_first"
    elif data_format == "NHWC":
        channel_data_format = "channels_last"
    else:
        raise ValueError("data format must be 'NCHW' or 'NHWC'. got {}.".format(data_format))

    with tf.variable_scope(name):
        conved = tf.layers.conv2d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  # he initializer
            data_format=channel_data_format,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay_rate),
        )

        batch_normed = tf.contrib.layers.batch_norm(
            conved,
            decay=batch_norm_decay,
            updates_collections=None,
            is_training=is_training,
            center=True,
            scale=True,
            data_format=data_format,
            )

        if activation:
            output = activation(batch_normed)
        else:
            output = batch_normed

        if enable_detail_summary:
            tf.summary.histogram('conv_output', conved)
            tf.summary.histogram('batch_norm_output', batch_normed)
            tf.summary.histogram('output', output)

        return output


def _densenet_conv_bn_act(
        name,
        inputs,
        growth_rate,
        bottleneck_rate,
        weight_decay_rate,
        is_training,
        activation,
        batch_norm_decay,
        data_format,
        enable_detail_summary,
):
    """Densenet block.

    In order to fast execute for quantization, use order of layers
    convolution -> batch norm -> activation instead of paper original's batch norm -> activation -> convolution.
    This is not `Dense block` called by original paper, this is the part of `Dense block`.
    """
    bottleneck_channel = growth_rate * bottleneck_rate

    with tf.variable_scope(name):

        output_1x1 = conv_bn_act(
            "bottleneck_1x1",
            inputs,
            filters=bottleneck_channel,
            kernel_size=1,
            weight_decay_rate=weight_decay_rate,
            is_training=is_training,
            activation=activation,
            batch_norm_decay=batch_norm_decay,
            data_format=data_format,
            enable_detail_summary=enable_detail_summary,
        )

        output_3x3 = conv_bn_act(
            "conv_3x3",
            output_1x1,
            filters=growth_rate,
            kernel_size=3,
            weight_decay_rate=weight_decay_rate,
            is_training=is_training,
            activation=activation,
            batch_norm_decay=batch_norm_decay,
            data_format=data_format,
            enable_detail_summary=enable_detail_summary,
        )

        if data_format == "NHWC":
            concat_axis = -1
        if data_format == "NCHW":
            concat_axis = 1

        output = tf.concat([inputs, output_3x3], axis=concat_axis)

        if enable_detail_summary:
            tf.summary.histogram('output', output)

    return output


def densenet_group(
        name,
        inputs,
        num_blocks,
        growth_rate,
        bottleneck_rate=4,
        weight_decay_rate=0.0,
        is_training=tf.constant(False),
        activation=None,
        batch_norm_decay=0.99,
        data_format="NHWC",
        enable_detail_summary=False,
):
    """Group of Densenet blocks.

    paper: https://arxiv.org/abs/1608.06993
    In the original paper, this method is called `Dense block` which consists of some 1x1 and 3x3 conv blocks
    which batch norm -> activation(relu) -> convolution(1x1) and batch norm -> activation -> convolution(3x3).
    But in this method, the order of each block change to convolution -> batch norm -> activation.

    Args:
        name (str): Block name, as scope name.
        inputs (tf.Tensor): Inputs.
        num_blocks (int): Number of dense blocks which consist of 1x1 and 3x3 conv.
        growth_rate (int): How many filters (out channel) to add each layer.
        bottleneck_rate (int): The factor to be calculated bottle-neck 1x1 conv output channel.
            `bottleneck_channel = growth_rate * bottleneck_rate`.
            The default value `4` is from original paper.
        weight_decay_rate (float): Number of L2 regularization be applied to convolution weights.
        is_training (tf.constant): Flag if training or not.
        activation (callable): Activation function.
        batch_norm_decay (float): Batch norm decay rate.
        enable_detail_summary (bool): Flag for summarize feature maps for each operation on tensorboard.
        data_format (string):  Format for inputs data. NHWC or NCHW.
    Returns:
        tf.Tensor: Output of current  block.
    """

    with tf.variable_scope(name):
        x = inputs
        for i in range(num_blocks):
            x = _densenet_conv_bn_act(
                "densenet_block_{}".format(i),
                x,
                growth_rate,
                bottleneck_rate,
                weight_decay_rate,
                is_training,
                activation,
                batch_norm_decay,
                data_format,
                enable_detail_summary,
            )

        return x
