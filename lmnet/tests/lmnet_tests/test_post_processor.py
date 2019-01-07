# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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

from lmnet.post_processor import (
    FormatYoloV2,
    ExcludeLowScoreBox,
    NMS,
)

# Apply reset_default_graph() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("reset_default_graph")


# TODO(wakisaka): Can only test output shape.
def test_format_yolov2_shape():
    image_size = [128, 96]
    batch_size = 2
    classes = range(8)
    anchors = [(0.1, 0.2), (1.2, 1.1)]

    post_process = FormatYoloV2(
        image_size=image_size,
        classes=classes,
        anchors=anchors,
        data_format="NCHW",
    )

    shape = (batch_size, len(anchors) * (len(classes) + 5), image_size[0]//32, image_size[1]//32)
    output = np.random.uniform(-2., 2., size=shape).astype(np.float32)

    y = post_process(output)["outputs"]

    expected_shape = (batch_size, len(anchors) * len(classes) * image_size[0]//32 * image_size[1]//32, 6)

    assert expected_shape == y.shape


def test_exclude_low_score_box():
    threshold = 0.35
    inputs = np.array([
        [
            [10, 11, 12, 13, 1, 0.1],
            [20, 21, 22, 23, 2, 0.2],

        ],
        [
            [30, 31, 32, 33, 3, 0.3],
            [40, 41, 42, 43, 4, 0.4],
        ],
        [
            [50, 51, 52, 53, 5, 0.5],
            [60, 61, 62, 63, 6, 0.6],
        ],
    ])

    expected_ys = [
        np.zeros([0, 6]),
        np.array([
            [40, 41, 42, 43, 4, 0.4],
        ]),
        np.array([
            [50, 51, 52, 53, 5, 0.5],
            [60, 61, 62, 63, 6, 0.6],
        ]),
    ]
    post_process = ExcludeLowScoreBox(
        threshold=threshold,
    )

    ys = post_process(inputs)["outputs"]

    for expected_y, y in zip(expected_ys, ys):
        assert np.allclose(expected_y, y)


def test_nms():
    iou_threshold = 0.4
    classes = range(5)
    per_class = True

    inputs = [
        np.array([
            [10, 11, 12, 13, 1, 0.1],
            [11, 12, 13, 14, 1, 0.2],
            [12, 13, 14, 15, 1, 0.3],
            [80, 81, 22, 23, 2, 0.2],
        ]),
        np.array([
            [80, 81, 22, 23, 2, 0.1],
            [30, 31, 32, 33, 3, 0.3],
            [80, 81, 22, 23, 3, 0.2],
            [30, 31, 32, 33, 4, 0.4],
        ]),
        np.array([
            [60, 61, 62, 63, 2, 0.6],
            [82, 22, 32, 32, 2, 0.7],
            [83, 23, 33, 33, 2, 0.6],
        ]),
    ]

    expected_ys = [
        np.array([
            [12, 13, 14, 15, 1, 0.3],
            [80, 81, 22, 23, 2, 0.2],
        ]),
        np.array([
            [80, 81, 22, 23, 2, 0.1],
            [30, 31, 32, 33, 3, 0.3],
            [80, 81, 22, 23, 3, 0.2],
            [30, 31, 32, 33, 4, 0.4],
        ]),
        np.array([
            [82, 22, 32, 32, 2, 0.7],
            [60, 61, 62, 63, 2, 0.6],
        ]),
    ]
    post_process = NMS(
        classes=classes,
        iou_threshold=iou_threshold,
        per_class=per_class,
    )

    ys = post_process(inputs)["outputs"]

    for expected_y, y in zip(expected_ys, ys):
        assert np.allclose(expected_y, y), (expected_y, y)


def test_nms_not_per_class():
    iou_threshold = 0.4
    classes = range(5)
    per_class = False

    inputs = [
        np.array([
            [10, 11, 12, 13, 1, 0.1],
            [11, 12, 13, 14, 1, 0.2],
            [12, 13, 14, 15, 1, 0.3],
            [80, 81, 22, 23, 2, 0.2],
        ]),
        np.array([
            [80, 81, 22, 23, 2, 0.1],
            [30, 31, 32, 33, 3, 0.3],
            [80, 81, 22, 23, 3, 0.2],
            [30, 31, 32, 33, 4, 0.4],
        ]),
        np.array([
            [60, 61, 62, 63, 2, 0.6],
            [82, 22, 32, 32, 2, 0.7],
            [83, 23, 33, 33, 2, 0.6],
        ]),
    ]

    expected_ys = [
        np.array([
            [12, 13, 14, 15, 1, 0.3],
            [80, 81, 22, 23, 2, 0.2],
        ]),
        np.array([
            [30, 31, 32, 33, 4, 0.4],
            [80, 81, 22, 23, 3, 0.2],
        ]),
        np.array([
            [82, 22, 32, 32, 2, 0.7],
            [60, 61, 62, 63, 2, 0.6],
        ]),
    ]
    post_process = NMS(
        classes=classes,
        iou_threshold=iou_threshold,
        per_class=per_class,
    )

    ys = post_process(inputs)["outputs"]

    for expected_y, y in zip(expected_ys, ys):
        assert np.allclose(expected_y, y), (expected_y, y)


def test_nms_max_output_size():
    iou_threshold = 0.4
    classes = range(5)
    per_class = False
    max_output_size = 1

    inputs = [
        np.array([
            [10, 11, 12, 13, 1, 0.1],
            [11, 12, 13, 14, 1, 0.2],
            [12, 13, 14, 15, 1, 0.3],
            [80, 81, 22, 23, 2, 0.2],
        ]),
        np.array([
            [80, 81, 22, 23, 2, 0.1],
            [30, 31, 32, 33, 3, 0.3],
            [80, 81, 22, 23, 3, 0.2],
            [30, 31, 32, 33, 4, 0.4],
        ]),
        np.array([
            [60, 61, 62, 63, 2, 0.6],
            [82, 22, 32, 32, 2, 0.7],
            [83, 23, 33, 33, 2, 0.6],
        ]),
    ]

    expected_ys = [
        np.array([
            [12, 13, 14, 15, 1, 0.3],
        ]),
        np.array([
            [30, 31, 32, 33, 4, 0.4],
        ]),
        np.array([
            [82, 22, 32, 32, 2, 0.7],
        ]),
    ]
    post_process = NMS(
        classes=classes,
        iou_threshold=iou_threshold,
        per_class=per_class,
        max_output_size=max_output_size,
    )

    ys = post_process(inputs)["outputs"]

    for expected_y, y in zip(expected_ys, ys):
        assert np.allclose(expected_y, y), (expected_y, y)


if __name__ == '__main__':
    test_format_yolov2_shape()
    test_exclude_low_score_box()
    test_nms()
    test_nms_not_per_class()
    test_nms_max_output_size()
