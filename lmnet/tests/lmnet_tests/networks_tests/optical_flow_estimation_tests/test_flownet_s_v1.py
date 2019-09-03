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
from easydict import EasyDict
import numpy as np
import pytest
import tensorflow as tf

from executor.train import start_training
from lmnet import environment
from lmnet.networks.optical_flow_estimation.flownet_s_v1 import FlowNetSV1
from lmnet.utils.executor import prepare_dirs

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


# TODO Maybe don't need this
def get_conv1d_output_shape(length, kernel_size, striding=1, padding='SAME'):
    if padding == 'SAME':
        return int(np.ceil(length / striding))
    else:
        return (length - kernel_size + 2 * padding) // striding + 1


def test_conv_bn_act():
    inputs_shape = (1, 384, 512, 6)
    filters = 64
    kernel_size = 7
    strides = 2
    # TODO Can I randomly assign rgb values to images?
    inputs_np = np.random.uniform(0., 1., size=inputs_shape).astype(np.float32)

    # TODO remove cpu
    with tf.device('/cpu:0'):
        inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)

        model = FlowNetSV1(
            data_format="NHWC"
        )

        output_default = model._conv_bn_act(
            "conv_bn_act_default",
            inputs,
            filters,
            True
        )

        output_custom = model._conv_bn_act(
            "conv_bn_act",
            inputs,
            filters,
            True,
            kernel_size=kernel_size,
            strides=strides
        )
        output_custom_h = get_conv1d_output_shape(inputs_shape[1], kernel_size, strides)
        output_custom_w = get_conv1d_output_shape(inputs_shape[2], kernel_size, strides)

        init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)
    output_default_np = sess.run(output_default)
    assert output_default_np.shape == (inputs_shape[0], inputs_shape[1], inputs_shape[2], filters)

    output_custom_np = sess.run(output_custom)
    assert output_custom_np.shape == (inputs_shape[0], output_custom_h, output_custom_w, filters)


def test_deconv():
    inputs_shape = (1, 6, 8, 1024)
    filters = 512
    # TODO Can I randomly assign rgb values to images?
    inputs_np = np.random.uniform(0., 1., size=inputs_shape).astype(np.float32)

    with tf.device('/cpu:0'):
        inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)

        model = FlowNetSV1(
            data_format="NHWC"
        )

        output_default = model._deconv("deconv_default", inputs, filters)

        init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)
    output_default_np = sess.run(output_default)
    assert output_default_np.shape == (inputs_shape[0], inputs_shape[1] * 2, inputs_shape[2] * 2, filters)


def test_downsample():
    inputs_shape = (1, 200, 400, 2)
    output_shape = (1, 100, 200, 2)
    # TODO Need to know how flow is normalized
    inputs_np = np.random.uniform(-1., 1., size=inputs_shape).astype(np.float32)

    inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)

    model = FlowNetSV1(
        data_format="NHWC"
    )

    output_default = model._downsample("downsample_default", inputs, [output_shape[1], output_shape[2]])

    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)
    output_default_np = sess.run(output_default)
    assert output_default_np.shape == output_shape


def test_average_endpoint_error():
    output = np.array([
        [
            [
                [1, 2],
                [3, 4],
                [5, 6]
            ],
            [
                [7, 8],
                [9, 10],
                [11, 12]
            ]
        ]
    ])

    labels = np.array([
        [
            [
                [2, 7],
                [5, 3],
                [6, 6]
            ],
            [
                [9, 0],
                [3, 10],
                [7, 14]
            ]
        ]
    ])

    tf.InteractiveSession()
    output_tensor = tf.convert_to_tensor(output, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

    model = FlowNetSV1(
        data_format="NHWC"
    )

    batch_size = output.shape[0]
    squared_difference = np.square(np.subtract(output, labels))
    squared_difference = np.sum(squared_difference, axis=3, keepdims=True)
    avg_epe_per_pixel = np.sqrt(squared_difference)
    expected_avg_epe = np.sum(avg_epe_per_pixel) / batch_size
    expected_avg_epe = expected_avg_epe.astype(dtype=np.float32)

    avg_epe = model._average_endpoint_error(output_tensor, labels_tensor)
    avg_epe_np = avg_epe.eval()
    assert np.all(avg_epe_np == expected_avg_epe)


def test_contractive_block():
    inputs_shape = (2, 384, 512, 6)
    inputs_np = np.random.uniform(0., 1., size=inputs_shape).astype(np.float32)
    expected_output_shaped_dict = {
        'conv2': (2, 96, 128, 128),
        'conv3_1': (2, 48, 64, 256),
        'conv4_1': (2, 24, 32, 512),
        'conv5_1': (2, 12, 16, 512),
        'conv6_1': (2, 6, 8, 1024),
    }

    inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)

    model = FlowNetSV1(
        data_format="NHWC"
    )

    output = model._contractive_block(inputs, True)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)
    output_dict = sess.run(output)
    for name, shape in expected_output_shaped_dict.items():
        assert shape == output_dict[name].shape


if __name__ == '__main__':
    test_conv_bn_act()
    test_deconv()
    test_downsample()
    test_average_endpoint_error()
    test_contractive_block()

