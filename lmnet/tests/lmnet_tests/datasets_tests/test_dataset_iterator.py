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
import numpy as np
import pytest

from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.image_folder import ImageFolderBase

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


class Dummy(ImageFolderBase):
    extend_dir = "dummy_classification"


def test_dataset_iterator_batch_size():
    batch_size = 8
    dataset = Dummy(subset="train", batch_size=batch_size)
    dataset_iterator = DatasetIterator(dataset)

    for i in range(0, 10):
        images, labels = next(dataset_iterator)
        assert images.shape[0] == batch_size
        assert labels.shape[0] == batch_size

    batch_size = 32
    dataset = Dummy(subset="train", batch_size=batch_size)
    dataset_iterator = DatasetIterator(dataset)

    for i in range(0, 10):
        images, labels = next(dataset_iterator)
        assert images.shape[0] == batch_size
        assert labels.shape[0] == batch_size


def test_dataset_iterator_batch_order():
    """Assert that data given by iterator is same whether enabele_prefetch ture or false."""

    batch_size = 8
    dataset = Dummy(subset="train", batch_size=batch_size)
    dataset_iterator = DatasetIterator(dataset, seed=10, enable_prefetch=False)
    prefetch_dataset_iterator = DatasetIterator(dataset, seed=10, enable_prefetch=True)

    for i in range(0, 30):
        images, labels = next(dataset_iterator)
        prefetch_images, prefetch_labels = next(prefetch_dataset_iterator)

        assert np.all(images == prefetch_images)
        assert np.all(labels == prefetch_labels)


if __name__ == '__main__':
    from lmnet import environment
    environment.setup_test_environment()
    test_dataset_iterator_batch_size()
    test_dataset_iterator_batch_order()
