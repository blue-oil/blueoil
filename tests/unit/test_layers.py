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
import tensorflow as tf

from blueoil.layers import max_pool_with_argmax
from blueoil.layers.experiment import max_unpool_with_argmax


def test_unpool():
    tf.InteractiveSession()

    # shape is (1, 4, 4, 2)
    raw_inputs = np.array([
        [
            [
                [1, 0],
                [2, 7],
                [3, 7],
                [4, 7],
            ],
            [
                [5, 7],
                [6, 7],
                [7, 7],
                [8, 7],
            ],
            [
                [9, 7],
                [10, 0],
                [11, 0],
                [12, 8],
            ],
            [
                [13, 7],
                [14, 7],
                [15, 8],
                [16, 0],
            ],
        ],
    ])
    print(raw_inputs.shape)
    inputs = tf.convert_to_tensor(raw_inputs, dtype=tf.float32)

    expect_pooled = np.array([
        [
            [
                [6, 7],
                [8, 7],
            ],
            [
                [14, 7],
                [16, 8],
            ],
        ],
    ])

    # flatten indices
    expect_indices = np.array([
        [
            [
                [10, 3],
                [14, 5],
            ],
            [
                [26, 17],
                [30, 23],
            ],
        ],
    ])

    pooled, indices = max_pool_with_argmax("pool", inputs, pool_size=2, strides=2)

    assert np.all(pooled.eval() == expect_pooled)

    assert np.all(indices.eval() == expect_indices)

    expect_unpooled = np.array([
        [
            [
                [0, 0],
                [0, 7],
                [0, 7],
                [0, 0],
            ],
            [
                [0, 0],
                [6, 0],
                [0, 0],
                [8, 0],
            ],
            [
                [0, 7],
                [0, 0],
                [0, 0],
                [0, 8],
            ],
            [
                [0, 0],
                [14, 0],
                [0, 0],
                [16, 0],
            ],
        ],
    ])

    unpooled = max_unpool_with_argmax(pooled, indices, (1, 2, 2, 1))

    assert np.all(unpooled.eval() == expect_unpooled)


if __name__ == '__main__':
    test_unpool()
