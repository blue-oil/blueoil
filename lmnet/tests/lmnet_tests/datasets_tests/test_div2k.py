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
import os
from glob import glob

import numpy as np
import pytest

from lmnet import environment
from lmnet.datasets.dataset_iterator import DatasetIterator
from lmnet.datasets.div2k import Div2k


@pytest.mark.parametrize("type", [
    {"subset": "train", "dir": "DIV2K_train_HR"},
    {"subset": "validation", "dir": "DIV2K_valid_HR"},
])
def test_files(set_test_environment, type):
    dataset = Div2k(type["subset"])
    files = sorted(dataset.files)

    dataset_files = os.path.join(environment.DATA_DIR, "DIV2K/{}/*.png".format(type["dir"]))
    expected = sorted(glob(dataset_files))

    assert files == expected


@pytest.mark.parametrize("subset", ["train", "validation"])
def test_length(set_test_environment, subset):
    dataset = Div2k(subset)

    assert len(dataset.files) == len(dataset)


@pytest.mark.parametrize("subset", ["train", "validation"])
def test_num_per_epoch(set_test_environment, subset):
    dataset = Div2k(subset)

    assert len(dataset.files) == dataset.num_per_epoch


@pytest.mark.parametrize("subset", ["train", "validation"])
def test_get_item(set_test_environment, subset):
    dataset = Div2k(subset)

    for i in range(len(dataset)):
        image, _ = dataset[i]
        assert isinstance(image, np.ndarray)


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
