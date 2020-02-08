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
import time

import numpy as np
import PIL.Image
import PIL.ImageDraw
import pytest

from blueoil.nn.datasets.pascalvoc_2007 import Pascalvoc2007
from lmnet.utils.image import load_image
from blueoil.nn.datasets.dataset_iterator import DatasetIterator
from lmnet.pre_processor import ResizeWithGtBoxes
from lmnet.data_processor import (
    Sequence
)
from lmnet.data_augmentor import (
    Blur,
    Brightness,
    Color,
    Contrast,
    Crop,
    FlipLeftRight,
    FlipTopBottom,
    Hue,
    Pad,
    RandomPatchCut,
    SSDRandomCrop
)
from lmnet.utils.box import iou, crop_boxes


# Apply reset_default_graph() and set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


# TODO(wakisaka): All the tests are only checking if they are just working.

IS_DEBUG = False


def _show_images_with_boxes(images, labels):
    if not IS_DEBUG:
        return
    """show image and bounding box for debug"""
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
        time.sleep(0.5)


def _show_image(image):
    if not IS_DEBUG:
        return
    """show image for debug"""
    image = PIL.Image.fromarray(image)
    image.show()
    time.sleep(1)


def _image():
    image = load_image("tests/fixtures/sample_images/cat.jpg")

    return image


def test_sequence():
    batch_size = 3
    image_size = [256, 512]
    augmentor = Sequence([
        FlipLeftRight(),
        FlipTopBottom(),
        SSDRandomCrop(),
    ])

    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=augmentor,
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_blur():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=Blur((0, 1)),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_brightness():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=Brightness(),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_color():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=Color((0.0, 2.0)),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_contrast():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=Contrast((0.0, 2.0)),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_crop():
    img = _image()
    augmentor = Crop(size=(100, 200), resize=(200, 200))

    result = augmentor(**{'image': img})
    image = result['image']

    _show_image(image)

    assert image.shape[0] == 100
    assert image.shape[1] == 200
    assert image.shape[2] == 3


def test_filp_left_right():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=FlipLeftRight(),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_filp_top_bottom():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=FlipTopBottom(),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_hue():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=Hue((-10, 10)),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)


def test_pad():
    img = _image()
    assert img.shape[0] == 480
    assert img.shape[1] == 480
    assert img.shape[2] == 3

    augmentor = Pad(100)

    result = augmentor(**{'image': img})
    image = result['image']

    _show_image(image)

    assert image.shape[0] == 680
    assert image.shape[1] == 680
    assert image.shape[2] == 3

    augmentor = Pad((40, 30))

    result = augmentor(**{'image': img})
    image = result['image']

    _show_image(image)

    assert image.shape[0] == 480 + 30 * 2
    assert image.shape[1] == 480 + 40 * 2
    assert image.shape[2] == 3


def test_random_patch_cut():
    img = _image()
    assert img.shape[0] == 480
    assert img.shape[1] == 480
    assert img.shape[2] == 3

    augmentor = RandomPatchCut(num_patch=10, max_size=10, square=True)
    result = augmentor(**{'image': img})
    image = result['image']
    assert image.shape[0] == 480
    assert image.shape[1] == 480
    assert image.shape[2] == 3

    _show_image(image)


def test_ssd_random_crop():
    batch_size = 3
    image_size = [256, 512]
    dataset = Pascalvoc2007(
        batch_size=batch_size, pre_processor=ResizeWithGtBoxes(image_size),
        augmentor=SSDRandomCrop(),
    )
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()
        _show_images_with_boxes(images, labels)
        assert np.all(labels[:, :, 2] <= 512)
        assert np.all(labels[:, :, 3] <= 256)


def test_iou():
    box = np.array([
        10, 20, 80, 60,
    ])

    boxes = np.array([
        [10, 20, 80, 60, ],
        [10, 20, 40, 30, ],
        [50, 50, 80, 60, ],
    ])

    expected = np.array([
        1,
        0.25,
        (40 * 30) / (80 * 60 * 2 - 40 * 30),
    ])

    ious = iou(boxes, box)

    assert np.allclose(ious, expected)


def test_crop_boxes():

    boxes = np.array([
        [10, 20, 80, 60, 100],
    ])

    # boxes include crop
    crop_rect = [30, 30, 5, 5]
    expected = np.array([
        [0, 0, 5, 5, 100],
    ])
    cropped = crop_boxes(boxes, crop_rect)
    assert np.allclose(cropped, expected)

    crop_rect = [80, 30, 100, 100]
    expected = np.array([
        [0, 0, 10, 50, 100],
    ])
    cropped = crop_boxes(boxes, crop_rect)
    assert np.allclose(cropped, expected)

    # crop include box
    crop_rect = [5, 5, 100, 100]
    expected = np.array([
        [5, 15, 80, 60, 100],
    ])
    cropped = crop_boxes(boxes, crop_rect)
    assert np.allclose(cropped, expected)

    # overlap
    crop_rect = [10, 20, 80, 60]
    expected = np.array([
        [0, 0, 80, 60, 100],
    ])
    cropped = crop_boxes(boxes, crop_rect)
    assert np.allclose(cropped, expected)

    # When crop rect is external boxes, raise error.
    crop_rect = [0, 0, 5, 100]
    with pytest.raises(ValueError):
        cropped = crop_boxes(boxes, crop_rect)

    crop_rect = [0, 0, 100, 5]
    with pytest.raises(ValueError):
        cropped = crop_boxes(boxes, crop_rect)

    crop_rect = [95, 0, 5, 5]
    with pytest.raises(ValueError):
        cropped = crop_boxes(boxes, crop_rect)

    crop_rect = [30, 85, 5, 5]
    with pytest.raises(ValueError):
        cropped = crop_boxes(boxes, crop_rect)


if __name__ == '__main__':
    test_sequence()
    test_blur()
    test_brightness()
    test_color()
    test_contrast()
    test_crop()
    test_filp_left_right()
    test_filp_top_bottom()
    test_hue()
    test_pad()
    test_random_patch_cut()
    test_ssd_random_crop()
    test_iou()
    test_crop_boxes()
