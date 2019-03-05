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
import pytest
import numpy as np
from lmnet.datasets.camvid import Camvid, CamvidCustom
from lmnet.datasets.dataset_iterator import DatasetIterator

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


class DummyCamvid(Camvid):
    extend_dir = "camvid"


class DummyCamvidCustom(CamvidCustom):
    extend_dir = "camvid_custom"
    validation_extend_dir = "camvid_custom"


class DummyCamvidCustomWithoutTestDataset(CamvidCustom):
    extend_dir = "camvid_custom"


def _show_image_with_annotation(image, label, colors):
    """show image and annotation for debug"""

    import PIL.Image
    import PIL.ImageDraw

    image = PIL.Image.fromarray(image)

    # Make annotation color image
    label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype='uint8')
    for h in range(label.shape[0]):
        for w in range(label.shape[1]):
            label_rgb[h, w] = np.array(colors[label[h, w]])
    label = PIL.Image.fromarray(label_rgb)

    image.show()
    label.show()


def _test_camvid_basics(train_dataset, test_dataset):
    # test training dataset
    train_image_files, train_label_files = train_dataset.feed()
    assert train_image_files.shape[0] == 1
    assert train_label_files.shape[0] == 1

    train_images, train_labels = train_dataset.feed()
    assert isinstance(train_images, np.ndarray)
    assert train_images.shape == (1, 360, 480, 3)
    assert train_labels.shape == (1, 360, 480)

    # _show_image_with_annotation(train_images[0], train_labels[0], colors)

    # test test dataset
    test_image_files, test_label_files = test_dataset.feed()
    assert test_image_files.shape[0] == 1
    assert test_label_files.shape[0] == 1

    test_images, test_labels = test_dataset.feed()
    assert isinstance(test_images, np.ndarray)
    assert test_images.shape == (1, 360, 480, 3)
    assert test_labels.shape == (1, 360, 480)

    # _show_image_with_annotation(test_images[0], test_labels[0], colors)


def test_camvid():
    batch_size = 1
    train_dataset = DummyCamvid(subset="train", batch_size=batch_size)
    train_dataset = DatasetIterator(train_dataset)
    test_dataset = DummyCamvid(subset="validation", batch_size=batch_size)
    test_dataset = DatasetIterator(test_dataset)

    assert train_dataset.num_classes == 11
    colors = train_dataset.label_colors
    assert len(colors) == 12

    _test_camvid_basics(train_dataset, test_dataset)


def test_camvid_custom():
    batch_size = 1
    train_dataset = DummyCamvidCustom(subset="train", batch_size=batch_size)
    train_dataset = DatasetIterator(train_dataset)
    test_dataset = DummyCamvidCustom(subset="validation", batch_size=batch_size)
    test_dataset = DatasetIterator(test_dataset)

    assert train_dataset.num_classes == 12
    colors = train_dataset.label_colors
    assert len(colors) == 12

    _test_camvid_basics(train_dataset, test_dataset)


def test_camvid_custom_without_test_dataset():
    batch_size = 5
    validation_size = 0.2

    train_dataset = DummyCamvidCustomWithoutTestDataset(subset="train", batch_size=batch_size,
                                                        validation_size=validation_size)
    train_dataset = DatasetIterator(train_dataset)

    test_dataset = DummyCamvidCustomWithoutTestDataset(subset="validation", batch_size=batch_size,
                                                       validation_size=validation_size)
    test_dataset = DatasetIterator(test_dataset)

    assert train_dataset.num_per_epoch == 5 * (1 - validation_size)
    assert test_dataset.num_per_epoch == 5 * (validation_size)

    image_files, label_files = train_dataset.feed()
    assert image_files.shape[0] == 5
    assert label_files.shape[0] == 5

    image_files, label_files = test_dataset.feed()
    assert image_files.shape[0] == 5
    assert label_files.shape[0] == 5
