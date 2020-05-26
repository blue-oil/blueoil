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

from blueoil.pre_processor import (
    Resize,
    DivideBy255,
    ResizeWithJoints,
    JointsToGaussianHeatmap
)

# Apply reset_default_graph() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("reset_default_graph")


def test_resize():
    IMAGE_SIZE = [32, 32]
    orig_image = np.zeros(shape=(1024, 512, 3), dtype=np.uint8)
    orig_mask = np.zeros(shape=(1024, 512, 3), dtype=np.uint8)

    pre_processor = Resize(IMAGE_SIZE)
    resized = pre_processor(image=orig_image, mask=orig_mask)
    resized_image = resized["image"]
    resized_mask = resized["mask"]

    assert isinstance(resized_image, np.ndarray)
    assert isinstance(resized_mask, np.ndarray)
    assert resized_image.shape[:2] == (32, 32)
    assert resized_image.shape[2] == 3

    assert resized_mask.shape[:2] == (32, 32)
    assert resized_mask.shape[2] == 3


def test_divide_255():
    orig_image = np.ones((1, 1, 3))
    expect = np.array([[[0.00392157, 0.00392157, 0.00392157]]])

    pre_processor = DivideBy255()
    processed = pre_processor(image=orig_image)
    processed_imaged = processed["image"]

    assert isinstance(processed_imaged, np.ndarray)
    assert processed_imaged.shape[:2] == orig_image.shape[:2]
    assert np.allclose(expect, processed_imaged)


def test_resize_with_joints():

    image = np.zeros(shape=(5, 5, 3), dtype=np.uint8)
    joints = np.ones(shape=(1, 3))

    # x
    joints[0, 0] = 2
    # y
    joints[0, 1] = 3

    image_size = (10, 10)
    resizer_10x10 = ResizeWithJoints(image_size)

    # No joints will be provided on inference time.
    resized = resizer_10x10(image=image)
    resized_image = resized["image"]

    assert resized_image.shape[0] == 10
    assert resized_image.shape[1] == 10
    assert resized_image.shape[2] == 3

    resized = resizer_10x10(image=image, joints=joints)
    resized_image = resized["image"]
    resized_joints = resized["joints"]

    assert isinstance(resized_image, np.ndarray)
    assert isinstance(resized_joints, np.ndarray)

    assert resized_image.shape[0] == 10
    assert resized_image.shape[1] == 10
    assert resized_image.shape[2] == 3

    assert resized_joints[0, 0] == 4
    assert resized_joints[0, 1] == 6
    assert resized_joints[0, 2] == joints[0, 2]

    # joints should not be changed in-place.
    assert resized_joints is not joints

    image_size = (10, 20)
    resizer_10x20 = ResizeWithJoints(image_size)

    # No joints will be provided on inference time.
    resized = resizer_10x20(image=image, joints=joints)
    resized_image = resized["image"]
    resized_joints = resized["joints"]

    assert resized_image.shape[0] == 10
    assert resized_image.shape[1] == 20
    assert resized_image.shape[2] == 3

    assert resized_joints[0, 0] == 8
    assert resized_joints[0, 1] == 6
    assert resized_joints[0, 2] == joints[0, 2]


def test_joints_to_gaussian_heatmap():

    image_size = (256, 320)

    stride = 2
    num_joints = 17

    input_joints = np.array([[1, 1, 1],
                             [2, 2, 1],
                             [3, 3, 1],
                             [4, 4, 1],
                             [5, 5, 1],
                             [6, 6, 1],
                             [7, 7, 1],
                             [8, 8, 1],
                             [9, 9, 1],
                             [10, 10, 1],
                             [11, 11, 1],
                             [12, 12, 1],
                             [13, 13, 1],
                             [14, 14, 1],
                             [15, 15, 1],
                             [16, 16, 1],
                             [17, 17, 0]])

    pre_process = JointsToGaussianHeatmap(image_size, num_joints=num_joints,
                                          stride=stride, sigma=2)

    heatmap = pre_process(joints=input_joints)["heatmap"]

    # It is hard to test semantic correctness of a gaussian heatmap manually.
    # That part will be tested jointly with GaussianHeatmapToJoints() in test_post_processor.py.
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape[0] == image_size[0] // stride
    assert heatmap.shape[1] == image_size[1] // stride
    assert heatmap.shape[2] == 17
    assert np.max(heatmap) == 10
