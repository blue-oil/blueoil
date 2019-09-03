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


if __name__ == '__main__':
    test_conv_bn_act()
