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

from lmnet.pre_processor import Resize, ResizeWithGtBoxes
from blueoil.datasets.open_images_v4 import OpenImagesV4BoundingBox
from blueoil.datasets.open_images_v4 import OpenImagesV4Classification
from blueoil.datasets.open_images_v4 import OpenImagesV4BoundingBoxBase
from blueoil.datasets.dataset_iterator import DatasetIterator

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_open_images_v4_classification():
    batch_size = 1
    image_size = [256, 256]
    dataset = OpenImagesV4Classification(batch_size=batch_size,
                                         pre_processor=Resize(image_size))
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == dataset.num_classes


def _show_images_with_boxes(images, labels):
    """show image for debug"""

    import PIL.Image
    import PIL.ImageDraw
    import time

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

        image.show()
        time.sleep(1.5)


def test_open_images_v4_object_detection():
    batch_size = 1
    image_size = [256, 256]
    dataset = OpenImagesV4BoundingBox(batch_size=batch_size,
                                      pre_processor=ResizeWithGtBoxes(image_size))
    dataset = DatasetIterator(dataset)

    num_max_boxes = dataset.num_max_boxes
    assert dataset.num_max_boxes == OpenImagesV4BoundingBox.count_max_boxes()

    for _ in range(5):
        images, labels = dataset.feed()

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


class Dummy(OpenImagesV4BoundingBoxBase):
    extend_dir = "custom_open_images_v4_bounding_boxes/for_train"


def test_custom_open_images_v4_object_detection():
    validation_size = 0.2
    batch_size = 1
    image_size = [256, 128]
    train_dataset = Dummy(batch_size=batch_size,
                          validation_size=validation_size,
                          pre_processor=ResizeWithGtBoxes(image_size))
    train_dataset = DatasetIterator(train_dataset)

    validation_dataset = Dummy(batch_size=batch_size,
                               subset="validation",
                               validation_size=validation_size,
                               pre_processor=ResizeWithGtBoxes(image_size))
    validation_dataset = DatasetIterator(validation_dataset)

    num_max_boxes = train_dataset.num_max_boxes
    assert train_dataset.num_max_boxes == Dummy.count_max_boxes()

    assert train_dataset.num_per_epoch == 10 * (1 - validation_size)
    assert validation_dataset.num_per_epoch == 10 * (validation_size)

    for _ in range(13):
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


class DummyHasValidation(OpenImagesV4BoundingBoxBase):
    extend_dir = "custom_open_images_v4_bounding_boxes/for_train"
    validation_extend_dir = "custom_open_images_v4_bounding_boxes/for_validation"


def test_custom_has_validation_open_images_v4_object_detection():
    batch_size = 8
    image_size = [196, 128]
    train_dataset = DummyHasValidation(subset="train", batch_size=batch_size,
                                       pre_processor=ResizeWithGtBoxes(image_size))
    train_dataset = DatasetIterator(train_dataset)
    validation_dataset = DummyHasValidation(subset="validation", batch_size=batch_size,
                                            pre_processor=ResizeWithGtBoxes(image_size))
    validation_dataset = DatasetIterator(validation_dataset)

    num_max_boxes = validation_dataset.num_max_boxes
    assert validation_dataset.num_max_boxes == DummyHasValidation.count_max_boxes()

    assert train_dataset.num_per_epoch == 10
    assert validation_dataset.num_per_epoch == 16
    assert len(train_dataset.classes) == 44
    assert len(validation_dataset.classes) == 44

    for _ in range(3):
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

    for _ in range(3):
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

    test_open_images_v4_classification()
    test_open_images_v4_object_detection()
    test_custom_open_images_v4_object_detection()
    test_custom_has_validation_open_images_v4_object_detection()
