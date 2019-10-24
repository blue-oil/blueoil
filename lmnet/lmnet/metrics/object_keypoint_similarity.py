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
import numpy as np


# kpt_oks_sigmas is from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35,
                           .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
variances = (kpt_oks_sigmas * 2) ** 2


def compute_object_keypoint_similarity(joints_gt, joints_pred, image_size=(160, 160)):
    """
    Compute a object keypoint similarity for a batch of examples.
    Args:
        joints_gt: a numpy array of shape (batch_size, num_joints, 3).
        joints_pred: a numpy array of shape (batch_size, num_joints, 3).
        image_size: a tuple, (height, width).

    Returns:
        oks_batch: float.

    """

    batch_size = joints_gt.shape[0]

    oks_batch = 0
    # count of valid joints in a batch
    count = 0

    for i in range(batch_size):
        oks = _compute_oks(joints_gt[i], joints_pred[i], image_size=image_size)
        if oks != -1:
            oks_batch += oks
            count += 1

    if count == 0:
        raise ValueError("Count of valid joint can not be zero.")

    oks_batch /= count

    return np.float32(oks_batch)


def _compute_oks(joints_gt, joints_pred, image_size=(160, 160)):
    """
    Compute a object keypoint similarity for one example.
    Args:
        joints_gt: a numpy array of shape (num_joints, 3).
        joints_pred: a numpy array of shape (num_joints, 3).
        image_size: a tuple, (height, width).

    Returns:
        oks: float.

    """

    num_joints = joints_gt.shape[0]

    x_gt = joints_gt[:, 0]
    y_gt = joints_gt[:, 1]
    # visibility of ground-truth joints
    v_gt = joints_gt[:, 2]

    x_pred = joints_pred[:, 0]
    y_pred = joints_pred[:, 1]

    area = image_size[0] * image_size[1]

    squared_distance = (x_gt - x_pred) ** 2 + (y_gt - y_pred) ** 2

    squared_distance /= (area * variances * 2)

    oks = 0
    count = 0

    for i in range(num_joints):
        if v_gt[i] > 0:
            oks += np.exp(-squared_distance[i], dtype=np.float32)
            count += 1

    if count == 0:
        return -1
    else:
        oks /= count
        return oks
