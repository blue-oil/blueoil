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
import base64
import json
from io import BytesIO

import numpy as np
import PIL.Image

from nn.common import Tasks
from nn.utils.predict_output.output import JsonOutput


def test_classification_json():
    task = Tasks.CLASSIFICATION
    image_size = (120, 160)
    classes = ("aaa", "bbb")
    params = {
        "task": task,
        "classes": classes,
        "image_size": image_size,
        "data_format": "NCHW",
    }

    batch_size = 2
    inputs = np.random.uniform(size=[batch_size, len(classes)])

    raw_images = np.zeros((batch_size, 320, 280, 3), dtype=np.uint8)
    image_files = ["dummy.png", "dumpy_2.pny"]

    call = JsonOutput(**params)

    json_output = call(inputs, raw_images, image_files)
    output = json.loads(json_output)

    assert output["classes"] == [{"id": i, "name": name} for i, name in enumerate(classes)]
    assert output["task"] == str(task.value)

    results = output["results"]
    assert [result["file_path"] for result in results] == image_files

    for i in range(batch_size):
        predictions = results[i]["prediction"]
        assert [prediction["probability"] for prediction in predictions] == inputs[i, :].astype(str).tolist()


def test_object_detection_json():
    task = Tasks.OBJECT_DETECTION
    image_size = (120, 160)
    classes = ("aaa", "bbb")
    params = {
        "task": task,
        "classes": classes,
        "image_size": image_size,
        "data_format": "NCHW",
    }

    batch_size = 2
    box_sizes = (3, 5)

    boxes_1 = np.concatenate([
        np.random.randint(120, size=(box_sizes[0], 4)),
        np.random.randint(len(classes), size=(box_sizes[0], 1)),
        np.random.uniform(size=(box_sizes[0], 1)),
    ], axis=1)

    boxes_2 = np.concatenate([
        np.random.randint(120, size=(box_sizes[1], 4)),
        np.random.randint(len(classes), size=(box_sizes[1], 1)),
        np.random.uniform(size=(box_sizes[1], 1)),
    ], axis=1)

    inputs = [boxes_1, boxes_2]

    raw_images = np.zeros((batch_size, 320, 280, 3), dtype=np.uint8)
    image_files = ["dummy.png", "dumpy_2.pny"]

    call = JsonOutput(**params)

    json_output = call(inputs, raw_images, image_files)
    output = json.loads(json_output)

    assert output["classes"] == [{"id": i, "name": name} for i, name in enumerate(classes)]
    assert output["task"] == str(task.value)

    results = output["results"]
    assert [result["file_path"] for result in results] == image_files

    for i in range(batch_size):
        predictions = results[i]["prediction"]
        assert [prediction["score"] for prediction in predictions] == inputs[i][:, 5].astype(str).tolist()
        assert [prediction["class"]["id"] for prediction in predictions] == inputs[i][:, 4].astype(int).tolist()

        resized_boxes = np.stack([
            inputs[i][:, 0] * 280 / image_size[1],
            inputs[i][:, 1] * 320 / image_size[0],
            inputs[i][:, 2] * 280 / image_size[1],
            inputs[i][:, 3] * 320 / image_size[0],
        ], axis=1)
        assert np.allclose([prediction["box"] for prediction in predictions], resized_boxes)


def test_semantic_segmentation_json():
    task = Tasks.SEMANTIC_SEGMENTATION
    image_size = (120, 160)
    classes = ("aaa", "bbb")
    params = {
        "task": task,
        "classes": classes,
        "image_size": image_size,
        "data_format": "NCHW",
    }

    batch_size = 2

    predict = np.random.uniform(size=(batch_size, len(classes), image_size[0], image_size[1]))

    raw_images = np.zeros((batch_size, 320, 280, 3), dtype=np.uint8)
    image_files = ["dummy.png", "dumpy_2.pny"]

    call = JsonOutput(**params)

    json_output = call(predict, raw_images, image_files)
    output = json.loads(json_output)

    assert output["classes"] == [{"id": i, "name": name} for i, name in enumerate(classes)]
    assert output["task"] == str(task.value)

    results = output["results"]
    assert [result["file_path"] for result in results] == image_files

    for i in range(batch_size):
        predictions = results[i]["prediction"]
        for class_id in range(len(classes)):
            mask = predictions[i]["mask"]
            mask_data = base64.b64decode(mask)
            mask_pil_image = PIL.Image.open(BytesIO(mask_data))
            mask_image = np.array(mask_pil_image)
            assert mask_image.shape == (320, 280)


if __name__ == '__main__':
    test_classification_json()
    test_object_detection_json()
    test_semantic_segmentation_json()
