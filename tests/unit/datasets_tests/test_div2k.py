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

from blueoil.datasets.dataset_iterator import DatasetIterator
from blueoil.datasets.div2k import Div2k


def test_train_files(set_test_environment):
    expected = [
        "unit/fixtures/datasets/DIV2K/DIV2K_train_HR/0001.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_train_HR/0002.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_train_HR/0003.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_train_HR/0004.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_train_HR/0005.png",
    ]
    assert sorted(Div2k("train").files) == expected


def test_validation_files(set_test_environment):
    expected = [
        "unit/fixtures/datasets/DIV2K/DIV2K_valid_HR/0001.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_valid_HR/0002.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_valid_HR/0003.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_valid_HR/0004.png",
        "unit/fixtures/datasets/DIV2K/DIV2K_valid_HR/0005.png",
    ]
    assert sorted(Div2k("validation").files) == expected


def test_length(set_test_environment):
    expected = 5
    assert len(Div2k("train")) == expected


def test_num_per_epoch(set_test_environment):
    expected = 5
    assert Div2k("train").num_per_epoch == expected


def test_get_item(set_test_environment):
    assert all(isinstance(image, np.ndarray) for image, _ in Div2k("train"))


@pytest.mark.parametrize("subset", ["train", "validation"])
def test_can_iterate(set_test_environment, subset):
    batch_size = 1
    image_size = (100, 100)

    dataset = Div2k(subset, batch_size=batch_size)
    iterator = DatasetIterator(dataset)

    for _ in range(len(dataset)):
        images, labels = iterator.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels[0] is None
