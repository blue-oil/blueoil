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
import pytest
import numpy as np

from blueoil.metrics.object_keypoint_similarity import compute_object_keypoint_similarity, _compute_oks


# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_compute_oks():

    # case1

    joints_gt = np.zeros((17, 3))
    joints_pred = np.zeros((17, 3))
    image_size = (160, 160)

    joints_gt[0, 0] = 80
    joints_gt[0, 1] = 80
    joints_gt[0, 2] = 1

    joints_pred[0, 0] = 70
    joints_pred[0, 1] = 70
    joints_pred[0, 2] = 1

    joints_pred[2, 0] = 1000
    joints_pred[2, 1] = 1000
    joints_pred[2, 2] = 1

    expected = 0.2358359

    result = _compute_oks(joints_gt, joints_pred, image_size)

    assert np.allclose(result, expected)

    # case2

    joints_gt = np.zeros((17, 3))
    joints_pred = np.zeros((17, 3))
    image_size = (160, 160)

    joints_gt[0, 0] = 80
    joints_gt[0, 1] = 80
    joints_gt[0, 2] = 0

    joints_pred[0, 0] = 70
    joints_pred[0, 1] = 70
    joints_pred[0, 2] = 1

    joints_pred[2, 0] = 1000
    joints_pred[2, 1] = 1000
    joints_pred[2, 2] = 1

    expected = -1

    result = _compute_oks(joints_gt, joints_pred, image_size)

    assert np.allclose(result, expected)

    # case3

    joints_gt = np.zeros((17, 3))
    joints_pred1 = np.zeros((17, 3))
    joints_pred2 = np.zeros((17, 3))
    image_size = (160, 160)

    joints_gt[0, 0] = 80
    joints_gt[0, 1] = 80
    joints_gt[0, 2] = 1

    joints_pred1[0, 0] = 70
    joints_pred1[0, 1] = 70
    joints_pred1[0, 2] = 1

    joints_pred2[0, 0] = 78
    joints_pred2[0, 1] = 78
    joints_pred2[0, 2] = 1

    result1 = _compute_oks(joints_gt, joints_pred1, image_size)
    result2 = _compute_oks(joints_gt, joints_pred2, image_size)

    assert result2 > result1


def test_compute_object_keypoint_similarity():

    # case1

    joints_gt = np.zeros((1, 17, 3))
    joints_pred = np.zeros((1, 17, 3))
    image_size = (160, 160)

    joints_gt[0, 0, 0] = 80
    joints_gt[0, 0, 1] = 80
    joints_gt[0, 0, 2] = 1

    joints_pred[0, 0, 0] = 70
    joints_pred[0, 0, 1] = 70
    joints_pred[0, 0, 2] = 1

    expected = 0.2358359

    result = compute_object_keypoint_similarity(joints_gt, joints_pred, image_size)

    assert np.allclose(result, expected)

    # case2

    joints_gt = np.zeros((2, 17, 3))
    joints_pred = np.zeros((2, 17, 3))
    image_size = (160, 160)

    joints_gt[0, 0, 0] = 80
    joints_gt[0, 0, 1] = 80
    joints_gt[0, 0, 2] = 1

    joints_pred[0, 0, 0] = 70
    joints_pred[0, 0, 1] = 70
    joints_pred[0, 0, 2] = 1

    joints_gt[1, 0, 0] = 50
    joints_gt[1, 0, 1] = 50
    joints_gt[1, 0, 2] = 1

    joints_pred[1, 0, 0] = 50
    joints_pred[1, 0, 1] = 50
    joints_pred[1, 0, 2] = 1

    expected = 0.61791795

    result = compute_object_keypoint_similarity(joints_gt, joints_pred, image_size)

    assert np.allclose(result, expected)

    # case3

    joints_gt = np.zeros((2, 17, 3))
    joints_pred = np.zeros((2, 17, 3))
    image_size = (160, 160)

    joints_gt[0, 0, 0] = 80
    joints_gt[0, 0, 1] = 80

    joints_pred[0, 0, 0] = 70
    joints_pred[0, 0, 1] = 70
    joints_pred[0, 0, 2] = 1

    joints_gt[1, 0, 0] = 50
    joints_gt[1, 0, 1] = 50

    joints_pred[1, 0, 0] = 50
    joints_pred[1, 0, 1] = 50
    joints_pred[1, 0, 2] = 1

    try:
        compute_object_keypoint_similarity(joints_gt, joints_pred, image_size)
    except ValueError:
        pass


if __name__ == '__main__':

    test_compute_oks()
    test_compute_object_keypoint_similarity()
