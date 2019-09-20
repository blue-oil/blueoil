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

from lmnet.datasets.cityscapes import Cityscapes
from lmnet.datasets.dataset_iterator import DatasetIterator

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_cityscapes():
    batch_size = 1
    train_dataset = Cityscapes(subset="train", batch_size=batch_size)
    train_dataset = DatasetIterator(train_dataset)

    test_dataset = Cityscapes(subset="validation", batch_size=batch_size)
    test_dataset = DatasetIterator(test_dataset)

    assert train_dataset.num_classes == 34
    colors = train_dataset.label_colors
    assert len(colors) == 34

    train_image_files, train_label_files = train_dataset.feed()
    assert train_image_files.shape[0] == batch_size
    assert train_label_files.shape[0] == batch_size
