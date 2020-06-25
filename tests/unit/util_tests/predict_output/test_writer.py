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
import json
import os

import numpy as np
import pytest
from PIL import Image

from blueoil.common import Tasks
from blueoil.utils.predict_output.writer import OutputWriter
from blueoil.utils.predict_output.writer import save_json
from blueoil.utils.predict_output.writer import save_npy
from blueoil.utils.predict_output.writer import save_materials


def test_write(temp_dir):
    task = Tasks.CLASSIFICATION
    classes = ("aaa", "bbb", "ccc")
    image_size = (320, 280)
    data_format = "NCHW"

    writer = OutputWriter(task, classes, image_size, data_format)
    outputs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    raw_images = np.zeros((3, 320, 280, 3), dtype=np.uint8)
    image_files = ["dummy1.png", "dummy2.png", "dummy3.png"]

    writer.write(temp_dir, outputs, raw_images, image_files, 1)

    assert os.path.exists(os.path.join(temp_dir, "npy", "1.npy"))
    assert os.path.exists(os.path.join(temp_dir, "json", "1.json"))
    assert os.path.exists(os.path.join(temp_dir, "images", "1", "aaa", "dummy3.png"))
    assert os.path.exists(os.path.join(temp_dir, "images", "1", "bbb", "dummy1.png"))
    assert os.path.exists(os.path.join(temp_dir, "images", "1", "ccc", "dummy2.png"))


def test_save_npy(temp_dir):
    """Test for save npy to existed dir"""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    save_npy(temp_dir, data, step=1)

    assert os.path.exists(os.path.join(temp_dir, "npy", "1.npy"))


def test_save_npy_not_existed_dir(temp_dir):
    """Test for save npy to not existed dir"""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    dist = os.path.join(temp_dir, 'not_existed')
    save_npy(dist, data, step=1)

    assert os.path.exists(os.path.join(dist, "npy", "1.npy"))


def test_save_npy_with_invalid_step(temp_dir):
    """Test for save npy with invalid step arg"""
    data = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError):
        save_npy(temp_dir, data, step={"invalid": "dict"})


def test_save_json(temp_dir):
    """Test for save json to existed dir"""
    data = json.dumps({"k": "v", "list": [1, 2, 3]})
    save_json(temp_dir, data, step=1)

    assert os.path.exists(os.path.join(temp_dir, "json", "1.json"))


def test_save_json_not_existed_dir(temp_dir):
    """Test for save json to not existed dir"""
    data = json.dumps({"k": "v", "list": [1, 2, 3]})
    dist = os.path.join(temp_dir, 'not_existed')
    save_json(dist, data, step=1)

    assert os.path.exists(os.path.join(dist, "json", "1.json"))


def test_save_json_with_invalid_step(temp_dir):
    """Test for save json with invalid step arg"""
    data = json.dumps({"k": "v", "list": [1, 2, 3]})

    with pytest.raises(ValueError):
        save_json(temp_dir, data, step={"invalid": "dict"})


def test_save_materials(temp_dir):
    """Test for save materials"""
    image1 = [[[0, 0, 0], [0, 0, 0]], [[255, 255, 255], [255, 255, 255]]]
    image2 = [[[0, 0, 0], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]]
    image3 = [[[255, 255, 255], [255, 255, 255]], [[0, 0, 0], [0, 0, 0]]]

    data = [
        ("image1.png", Image.fromarray(np.array(image1, dtype=np.uint8))),
        ("image2.png", Image.fromarray(np.array(image2, dtype=np.uint8))),
        ("image3.png", Image.fromarray(np.array(image3, dtype=np.uint8))),
    ]
    save_materials(temp_dir, data, step=1)

    assert os.path.exists(os.path.join(temp_dir, "images", "1", "image1.png"))
    assert os.path.exists(os.path.join(temp_dir, "images", "1", "image2.png"))
    assert os.path.exists(os.path.join(temp_dir, "images", "1", "image3.png"))


def test_save_materials_not_existed_dir(temp_dir):
    """Test for save materials to not existed dir"""
    image1 = [[[0, 0, 0], [0, 0, 0]], [[255, 255, 255], [255, 255, 255]]]
    image2 = [[[0, 0, 0], [255, 255, 255]], [[255, 255, 255], [0, 0, 0]]]
    image3 = [[[255, 255, 255], [255, 255, 255]], [[0, 0, 0], [0, 0, 0]]]

    data = [
        ("image1.png", Image.fromarray(np.array(image1, dtype=np.uint8))),
        ("image2.png", Image.fromarray(np.array(image2, dtype=np.uint8))),
        ("image3.png", Image.fromarray(np.array(image3, dtype=np.uint8))),
    ]
    dist = os.path.join(temp_dir, 'not_existed')
    save_materials(dist, data, step=1)

    assert os.path.exists(os.path.join(dist, "images", "1", "image1.png"))
    assert os.path.exists(os.path.join(dist, "images", "1", "image2.png"))
    assert os.path.exists(os.path.join(dist, "images", "1", "image3.png"))
