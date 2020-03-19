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

from blueoil.tfds_augmentor import (
    TFPad,
    TFCrop,
)

pytestmark = pytest.mark.usefixtures("reset_default_graph")

def test_tf_pad():
    orig_image_size = 512
    padding = 16
    orig_image = tf.zeros((orig_image_size, orig_image_size, 3), dtype=tf.dtypes.uint8)

    augmentor = TFPad(padding)
    result = augmentor(image=orig_image)
    padded_image = result["image"]
    with tf.Session() as sess:
        padded_image = sess.run(padded_image)

    assert isinstance(padded_image, np.ndarray)
    assert padded_image.shape[:2] == (orig_image_size + 2 * padding, orig_image_size + 2 * padding)
    assert padded_image.shape[2] == 3

def test_tf_crop():
    IMAGE_SIZE = [32, 32]
    orig_image = tf.zeros((1024, 512, 3), dtype=tf.dtypes.uint8)

    augmentor = TFCrop(IMAGE_SIZE)
    cropped = augmentor(image=orig_image)
    cropped_image = cropped["image"]
    with tf.Session() as sess:
        cropped_image = sess.run(cropped_image)

    assert isinstance(cropped_image, np.ndarray)
    assert cropped_image.shape[:2] == (32, 32)
    assert cropped_image.shape[2] == 3

if __name__ == '__main__':
    tf.reset_default_graph()
    test_tf_pad()
    test_tf_crop()