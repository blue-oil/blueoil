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
import PIL.Image

from lmnet.visualize import (
    draw_fps,
    visualize_classification,
    visualize_object_detection,
    visualize_semantic_segmentation,
)


def test_draw_fps():
    """Verify just image is changed."""
    pil_image = PIL.Image.new("RGB", size=(100, 200))
    stored = np.array(pil_image)
    fps = 11.1
    fps_only_network = 22.2

    draw_fps(pil_image, fps, fps_only_network)

    assert not np.all(np.array(stored) == np.array(pil_image))


def test_classification():
    """Verify just image is changed."""
    input_image = PIL.Image.new("RGB", size=(100, 200))
    results = np.array([0.1, 0.3, 0.4, 0.2])
    config = EasyDict({"CLASSES": ["a", "b", "c", "d"]})

    result_image = visualize_classification(np.array(input_image), results, config)

    assert not np.all(np.array(input_image) == np.array(result_image))


def test_object_detection():
    """Verify just image is changed."""
    input_image = PIL.Image.new("RGB", size=(100, 200))
    results = np.array([[32, 20, 10, 5, 2, 0.5], [2, 4, 2, 4, 1, 0.5]])
    config = EasyDict({"IMAGE_SIZE": (64, 64), "CLASSES": ["a", "b", "c", "d"]})

    result_image = visualize_object_detection(np.array(input_image), results, config)

    assert not np.all(np.array(input_image) == np.array(result_image))


def test_semantic_segmentation():
    """Verify just image is changed."""
    input_image = PIL.Image.new("RGB", size=(100, 200))
    results = np.random.random_sample(size=(64, 64, 4))
    config = EasyDict({"IMAGE_SIZE": (64, 64), "CLASSES": ["a", "b", "c", "d"]})

    result_image = visualize_semantic_segmentation(np.array(input_image), results, config)

    assert not np.all(np.array(input_image) == np.array(result_image))


if __name__ == '__main__':
    test_draw_fps()
    test_classification()
    test_object_detection()
    test_semantic_segmentation()
