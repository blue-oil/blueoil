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
import numpy as np
import pytest
import tensorflow as tf

from blueoil.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
    twn_weight_quantizer,
    ttq_weight_quantizer,
)

# Apply reset_default_graph() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("reset_default_graph")

default_x_size = (10, 2, 2, 4)


def numerical_derivative(x, fun):
    """Numerical derivative
    """
    h = 1e-5

    # use float64 to take care about derivative result precision.
    x = x.astype(np.float64)
    d = (fun(x + h) - fun(x - h)) / (2 * h)
    return d.astype(np.float32)


def make_grad_y(x_size=default_x_size):
    """Make grad_y

    """
    input_range = (-10., 10.)
    return np.random.uniform(input_range[0], input_range[1], size=x_size).astype(np.float32)


def make_test_input(x_size=default_x_size, input_range=(-10., 10.)):
    """Make test input
    """
    assert len(x_size) == 4
    assert len(input_range) == 2
    np_x = np.random.uniform(input_range[0], input_range[1], size=x_size).astype(np.float32)
    x = tf.convert_to_tensor(np_x)
    return np_x, x


def allclose(a, b):
    return np.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_binary_channel_wise_mean_scaling_quantizer():
    tf.InteractiveSession()

    quantizer = binary_channel_wise_mean_scaling_quantizer()

    def forward_np(x):
        expectation = np.mean(np.abs(np_x), axis=(0, 1, 2))
        return np.sign(x) * expectation

    def approximate_forward_np(x):
        return x

    np_x, x = make_test_input()
    grad_y = make_grad_y()

    expected_y = forward_np(np_x)
    expected_grad_x = grad_y * numerical_derivative(np_x, approximate_forward_np)

    y = quantizer(x)
    grad_x, = tf.gradients(y, x, grad_ys=grad_y)

    for split in np.split(y.eval(), 4, axis=3):
        assert len(np.unique(split)) == 2

    assert allclose(y.eval(), expected_y)
    assert allclose(grad_x.eval(), expected_grad_x)


def test_binary_mean_scaling_quantizer():
    tf.InteractiveSession()

    quantizer = binary_mean_scaling_quantizer()

    def forward_np(x):
        expectation = np.mean(np.abs(np_x))
        return np.sign(x / expectation) * expectation

    def approximate_forward_np(x):
        return x

    np_x, x = make_test_input()
    grad_y = make_grad_y()

    expected_y = forward_np(np_x)
    expected_grad_x = grad_y * numerical_derivative(np_x, approximate_forward_np)

    y = quantizer(x)
    grad_x, = tf.gradients(y, x, grad_ys=grad_y)

    assert allclose(y.eval(), expected_y)
    assert allclose(grad_x.eval(), expected_grad_x)


@pytest.mark.parametrize("bit_size", [2, 3])
@pytest.mark.parametrize("max_value", [1.0, 2.0])
def test_linear_mid_tread_half_quantizer(bit_size, max_value):
    tf.InteractiveSession()

    quantizer = linear_mid_tread_half_quantizer(bit=bit_size, max_value=max_value)

    min_value = 0.0

    def forward_np(x):
        n = float(2 ** bit_size - 1)
        value_range = max_value - min_value

        x = np.clip(x, min_value, max_value)
        shifted = (x - min_value) / value_range
        quantized = np.round(shifted * n) / n
        unshifted = quantized * value_range + min_value
        return unshifted

    def approximate_forward_np(x):
        any_const = 1
        return np.where((x < max_value) & (x > min_value), x, any_const)

    np_x, x = make_test_input()
    grad_y = make_grad_y()

    expected_y = forward_np(np_x)
    expected_grad_x = grad_y * numerical_derivative(np_x, approximate_forward_np)

    y = quantizer(x)
    grad_x, = tf.gradients(y, x, grad_ys=grad_y)

    assert allclose(y.eval(), expected_y)
    assert allclose(grad_x.eval(), expected_grad_x)


@pytest.mark.parametrize("threshold", [0.3, 0.7])
def test_twn_weight_quantizer(threshold):
    tf.InteractiveSession()

    quantizer = twn_weight_quantizer(threshold=threshold)

    def forward_np(weights):
        ternary_threshold = np.sum(np.abs(weights)) * threshold / np.size(weights)
        mask_positive = (weights > ternary_threshold)
        mask_negative = (weights < -ternary_threshold)
        mask_p_or_n = mask_positive | mask_negative

        p_or_n_weights = np.where(mask_p_or_n, weights, np.zeros_like(weights))
        scaling_factor = np.sum(np.abs(p_or_n_weights)) / np.sum(mask_p_or_n)

        positive_weights = scaling_factor * np.where(mask_positive, np.ones_like(weights), np.zeros_like(weights))
        negative_weights = - scaling_factor * np.where(mask_negative, np.ones_like(weights), np.zeros_like(weights))

        quantized = positive_weights + negative_weights

        return quantized

    def approximate_forward_np(weights):
        return weights

    np_x, x = make_test_input()
    grad_y = make_grad_y()

    expected_y = forward_np(np_x)
    expected_grad_x = grad_y * numerical_derivative(np_x, approximate_forward_np)

    y = quantizer(x)
    grad_x, = tf.gradients(y, x, grad_ys=grad_y)

    assert allclose(y.eval(), expected_y)
    assert allclose(grad_x.eval(), expected_grad_x)


# TODO(wakisaka): Test positive, negative is not 1.0 case. current these init by 1.0.
# TTP can't represent approximate forward.
def test_ttq_weight_quantizer():
    sess = tf.InteractiveSession()
    threshold = 0.005

    np_x = np.array([-5, -3, -0.0001, 0.0001, 5], dtype=np.float32)
    x = tf.convert_to_tensor(np_x)
    grad_y = tf.convert_to_tensor(np.array([1, 2, 3, 4, 5], dtype=np.float32))

    expected_y = np.array([-1, -1, 0, 0, 1], dtype=np.float32)
    expected_grad_x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    expected_grad_p = 5
    expected_grad_n = 1 + 2

    weight_quantizer = ttq_weight_quantizer(threshold)
    y = weight_quantizer(x)

    positive = tf.get_collection(tf.GraphKeys.VARIABLES, "positive")[0]
    negative = tf.get_collection(tf.GraphKeys.VARIABLES, "negative")[0]
    grad_x, grad_p, grad_n = tf.gradients(y, [x, positive, negative], grad_ys=grad_y)

    sess.run(tf.global_variables_initializer())

    assert allclose(y.eval(), expected_y)
    assert allclose(grad_x.eval(), expected_grad_x)
    assert allclose(grad_p.eval(), expected_grad_p)
    assert allclose(grad_n.eval(), expected_grad_n)


if __name__ == '__main__':
    test_binary_channel_wise_mean_scaling_quantizer()
    test_binary_mean_scaling_quantizer()
    test_linear_mid_tread_half_quantizer(2, 1.0)
    test_ttq_weight_quantizer()
    test_twn_weight_quantizer(0.33)
    pass
