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
import imghdr
import os
from glob import glob

import numpy as np
import pytest

from blueoil import environment
from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.image_folder import ImageFolderBase
from blueoil.pre_processor import Resize

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


class Dummy(ImageFolderBase):
    extend_dir = "dummy_classification"


def setup_dataset(dataset_class, subset, **kwargs):
    dataset = dataset_class(subset=subset, **kwargs)
    return DatasetIterator(dataset, seed=0)


def test_image_folder():
    validation_size = 0.2
    batch_size = 3
    image_size = [256, 256]
    train_dataset = setup_dataset(Dummy, subset="train",
                                  validation_size=validation_size,
                                  batch_size=batch_size,
                                  pre_processor=Resize(image_size))

    validation_dataset = setup_dataset(Dummy, subset="validation",
                                       validation_size=validation_size,
                                       batch_size=batch_size,
                                       pre_processor=Resize(image_size))

    expected_image_dir = os.path.join(environment.DATA_DIR, Dummy.extend_dir)
    expected_paths = [image_path for image_path in glob(os.path.join(expected_image_dir, "**/*"))
                      if os.path.isfile(image_path) and imghdr.what(image_path) in {"jpeg", "png"}]

    assert len(expected_paths) * (1-validation_size) == train_dataset.num_per_epoch
    assert len(expected_paths) * (validation_size) == validation_dataset.num_per_epoch

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


class DummyHasValidation(ImageFolderBase):
    extend_dir = "dummy_classification"
    validation_extend_dir = "open_images_v4"


def test_has_validation_path():
    batch_size = 3
    train_dataset = DummyHasValidation(subset="train",
                                       batch_size=batch_size)

    validation_dataset = DummyHasValidation(subset="validation",
                                            batch_size=batch_size)

    expected_train_image_dir = os.path.join(environment.DATA_DIR, DummyHasValidation.extend_dir)
    expected_train_paths = [image_path for image_path in glob(os.path.join(expected_train_image_dir, "**/*"))
                            if os.path.isfile(image_path) and imghdr.what(image_path) in {"jpeg", "png"}]

    assert len(expected_train_paths) == train_dataset.num_per_epoch

    expected_validation_image_dir = os.path.join(environment.DATA_DIR, DummyHasValidation.validation_extend_dir)
    expected_validation_paths = [image_path for image_path in glob(os.path.join(expected_validation_image_dir, "**/*"))
                                 if os.path.isfile(image_path) and imghdr.what(image_path) in {"jpeg", "png"}]

    assert len(expected_validation_paths) == validation_dataset.num_per_epoch


def test_image_folder_onthefly():
    Onthefly = type('Onthefly',
                    (ImageFolderBase,),
                    {"extend_dir": "dummy_classification", "validation_extend_dir": "open_images_v4"})
    batch_size = 3

    train_dataset = Onthefly(subset="train",
                             batch_size=batch_size)

    validation_dataset = Onthefly(subset="validation",
                                  batch_size=batch_size)

    expected_train_image_dir = os.path.join(environment.DATA_DIR, DummyHasValidation.extend_dir)
    expected_train_paths = [image_path for image_path in glob(os.path.join(expected_train_image_dir, "**/*"))
                            if os.path.isfile(image_path) and imghdr.what(image_path) in {"jpeg", "png"}]

    assert len(expected_train_paths) == train_dataset.num_per_epoch

    expected_validation_image_dir = os.path.join(environment.DATA_DIR, DummyHasValidation.validation_extend_dir)
    expected_validation_paths = [image_path for image_path in glob(os.path.join(expected_validation_image_dir, "**/*"))
                                 if os.path.isfile(image_path) and imghdr.what(image_path) in {"jpeg", "png"}]

    assert len(expected_validation_paths) == validation_dataset.num_per_epoch


if __name__ == '__main__':
    from blueoil.environment import setup_test_environment
    setup_test_environment()

    test_image_folder()
    test_has_validation_path()
    test_image_folder_onthefly()
