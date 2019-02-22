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

from lmnet.datasets.dataset_iterator import DatasetIterator
from lmnet.pre_processor import Resize, ResizeWithGtBoxes
from lmnet.datasets.delta_mark import (
    ClassificationBase,
    ObjectDetectionBase,
)


# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


def setup_dataset(dataset_class, subset, **kwargs):
    dataset = dataset_class(subset=subset, **kwargs)
    return DatasetIterator(dataset, seed=0)


def _show_images_with_boxes(images, labels):
    """show image for debug"""
    import time
    import PIL.Image
    import PIL.ImageDraw

    images_min = abs(images.min())
    images_max = (images + images_min).max()

    images = (images + images_min) * (255 / images_max)
    images = (images).astype(np.uint8)

    for image, label in zip(images, labels):
        image = PIL.Image.fromarray(image)
        draw = PIL.ImageDraw.Draw(image)

        for box in label:
            xy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            draw.rectangle(xy)

        print("image show")
        image.show()
        time.sleep(0.5)


class Dummy(ClassificationBase):
    extend_dir = "custom_delta_mark_classification/for_train"


def test_delta_mark_classification():
    validation_size = 1/3
    batch_size = 3
    image_size = [256, 128]
    all_data_num = 3

    train_dataset = setup_dataset(Dummy,
                                  subset="train",
                                  validation_size=validation_size,
                                  batch_size=batch_size,
                                  pre_processor=Resize(image_size))

    validation_dataset = setup_dataset(Dummy,
                                       subset="validation",
                                       validation_size=validation_size,
                                       batch_size=batch_size,
                                       pre_processor=Resize(image_size))

    assert train_dataset.num_per_epoch == (1 - validation_size) * all_data_num
    assert validation_dataset.num_per_epoch == validation_size * all_data_num

    for _ in range(5):
        images, labels = train_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == train_dataset.num_classes


class DummyHasValidation(ClassificationBase):
    extend_dir = "custom_delta_mark_classification/for_train"
    validation_extend_dir = "custom_delta_mark_classification/for_validation"


def test_delta_mark_classification_has_validation_path():
    batch_size = 3
    image_size = [256, 128]
    train_data_num = 3
    validation_data_num = 2

    train_dataset = setup_dataset(DummyHasValidation,
                                  subset="train",
                                  batch_size=batch_size,
                                  pre_processor=Resize(image_size))

    validation_dataset = setup_dataset(DummyHasValidation,
                                       subset="validation",
                                       batch_size=batch_size,
                                       pre_processor=Resize(image_size))

    assert train_dataset.num_per_epoch == train_data_num
    assert validation_dataset.num_per_epoch == validation_data_num

    for _ in range(5):
        images, labels = train_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == train_dataset.num_classes

    for _ in range(5):
        images, labels = validation_dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == validation_dataset.num_classes


class DummyObjectDetectio(ObjectDetectionBase):
    extend_dir = "custom_delta_mark_object_detection/for_train"


def test_delta_mark_object_detection():
    validation_size = 1/3
    batch_size = 3
    image_size = [256, 128]
    all_data_num = 3

    train_dataset = setup_dataset(DummyObjectDetectio,
                                  subset="train",
                                  validation_size=validation_size,
                                  batch_size=batch_size,
                                  pre_processor=ResizeWithGtBoxes(image_size))

    validation_dataset = setup_dataset(DummyObjectDetectio,
                                       subset="validation",
                                       validation_size=validation_size,
                                       batch_size=batch_size,
                                       pre_processor=ResizeWithGtBoxes(image_size))

    num_max_boxes = train_dataset.num_max_boxes
    assert train_dataset.num_max_boxes == DummyObjectDetectio.count_max_boxes()
    assert validation_dataset.num_max_boxes == DummyObjectDetectio.count_max_boxes()

    assert train_dataset.num_per_epoch == (1 - validation_size) * all_data_num
    assert validation_dataset.num_per_epoch == validation_size * all_data_num

    for _ in range(2):
        images, labels = train_dataset.feed()
        _show_images_with_boxes(images, labels)

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


class DummyObjectDetectionHasValidationPath(ObjectDetectionBase):
    extend_dir = "custom_delta_mark_object_detection/for_train"
    validation_extend_dir = "custom_delta_mark_object_detection/for_validation"


def test_delta_mark_object_detection_has_validation_path():
    batch_size = 4
    image_size = [256, 128]
    train_data_num = 3
    validation_data_num = 2

    train_dataset = setup_dataset(DummyObjectDetectionHasValidationPath,
                                  subset="train",
                                  batch_size=batch_size,
                                  pre_processor=ResizeWithGtBoxes(image_size))

    validation_dataset = setup_dataset(DummyObjectDetectionHasValidationPath,
                                       subset="validation",
                                       batch_size=batch_size,
                                       pre_processor=ResizeWithGtBoxes(image_size))

    num_max_boxes = train_dataset.num_max_boxes
    assert train_dataset.num_max_boxes == DummyObjectDetectionHasValidationPath.count_max_boxes()
    assert validation_dataset.num_max_boxes == DummyObjectDetectionHasValidationPath.count_max_boxes()

    assert train_dataset.num_per_epoch == train_data_num
    assert validation_dataset.num_per_epoch == validation_data_num

    for _ in range(2):
        images, labels = train_dataset.feed()
        # _show_images_with_boxes(images, labels)

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5

    for _ in range(2):
        images, labels = validation_dataset.feed()
        # _show_images_with_boxes(images, labels)

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_max_boxes
        assert labels.shape[2] == 5


if __name__ == '__main__':
    from lmnet.environment import setup_test_environment
    setup_test_environment()

    test_delta_mark_classification()
    test_delta_mark_classification_has_validation_path()

    test_delta_mark_object_detection()
    test_delta_mark_object_detection_has_validation_path()
