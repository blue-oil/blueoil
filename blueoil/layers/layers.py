#!/usr/bin/env python3
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
from functools import reduce

import tensorflow as tf


def batch_norm(
        name,
        inputs,
        is_training=tf.constant(False),
        activation=None,
        scale=True,
        *args,
        **kwargs
):
    if "data_format" in kwargs:
        axis = -1 if kwargs["data_format"] == 'NHWC' else 1
        kwargs.pop("data_format")
    else:
        axis = -1

    output = tf.compat.v1.layers.batch_normalization(
        inputs,
        axis=axis,
        training=is_training,
        name=name,
        scale=scale,
        *args,
        **kwargs,
    )

    if activation:
        output = activation(output)

    return output


def conv2d(
    name,
    inputs,
    filters,
    kernel_size,
    strides=1,
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform"),
    is_debug=False,
    *args,
    **kwargs
):

    output = tf.compat.v1.layers.conv2d(
        name=name,
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        *args,
        **kwargs,
    )

    if is_debug:
        tf.compat.v1.summary.histogram(name + "/output", output)

    return output


def fully_connected(
    name,
    inputs,
    filters,
    is_debug=False,
    activation=tf.nn.relu,
    *args,
    **kwargs
):
    # TODO(wakisaka): Bug. The condition `tf.rank(inputs) != 2` allways return True.
    # reshape
    if tf.rank(inputs) != 2:
        shape = inputs.get_shape().as_list()
        flattened_shape = reduce(lambda x, y: x*y, shape[1:])  # shp[1].value * shp[2].value * shp[3].value
        inputs = tf.reshape(inputs, [-1, flattened_shape], name=name + "_reshape")

    output = tf.contrib.layers.fully_connected(
        scope=name,
        inputs=inputs,
        num_outputs=filters,
        activation_fn=activation,
        *args,
        **kwargs,
    )

    if is_debug:
        tf.compat.v1.summary.histogram(name + "/output", output)

    return output


def max_pooling2d(
    name,
    inputs,
    pool_size,
    strides=1,
    padding="SAME",
    is_debug=False,
    *args,
    **kwargs
):

    output = tf.compat.v1.layers.max_pooling2d(
        name=name,
        inputs=inputs,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        *args,
        **kwargs,
    )

    if is_debug:
        tf.compat.v1.summary.histogram(name + "/output", output)

    return output


def max_pool_with_argmax(
    name,
    inputs,
    pool_size,
    strides=1,
    padding="SAME",
    is_debug=False,
    *args,
    **kwargs
):

    output = tf.nn.max_pool_with_argmax(
        inputs, ksize=[1, pool_size, pool_size, 1], strides=[1, strides, strides, 1], padding=padding, name=name,
    )

    return output


def average_pooling2d(
    name,
    inputs,
    pool_size,
    strides=1,
    padding="SAME",
    is_debug=False,
    *args,
    **kwargs
):

    output = tf.compat.v1.layers.average_pooling2d(
        name=name,
        inputs=inputs,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        *args,
        **kwargs
    )

    if is_debug:
        tf.compat.v1.summary.histogram(name + "/output", output)

    return output
