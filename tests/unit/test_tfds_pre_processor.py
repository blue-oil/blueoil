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
import pytest

from blueoil.tfds_pre_processor import (
    TFResize,
    TFResizeWithGtBoxes
)

pytestmark = pytest.mark.usefixtures("reset_default_graph")

def test_tf_resize():
    image_size = [32, 32]
    orig_image = tf.zeros((1024, 512, 3), dtype=tf.dtypes.uint8)

    pre_processor = TFResize(image_size)
    resized = pre_processor(image=orig_image)
    resized_image = resized["image"]
    with tf.Session() as sess:
        resized_image = sess.run(resized_image)

    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape[:2] == (32, 32)
    assert resized_image.shape[2] == 3

def test_tf_resize_with_gt_boxes():
    image_size = [32, 32]
    num_gt_boxes = 10
    orig_image = tf.zeros((1024, 1024, 3), dtype=tf.dtypes.uint8)
    gt_boxes = tf.zeros((num_gt_boxes, 5))

    pre_processor = TFResizeWithGtBoxes(image_size)
    resized = pre_processor(image=orig_image, gt_boxes=gt_boxes)
    with tf.Session() as sess:
        resized_image, resized_gt_boxes = sess.run([resized["image"], resized["gt_boxes"]])

    assert isinstance(resized_image, np.ndarray)
    assert resized_image.shape[:2] == (32, 32)
    assert resized_image.shape[2] == 3

    assert isinstance(resized_gt_boxes, np.ndarray)
    assert resized_gt_boxes.shape == (num_gt_boxes, 5)

if __name__ == '__main__':
    tf.reset_default_graph()
    test_tf_resize()
    test_tf_resize_with_gt_boxes()