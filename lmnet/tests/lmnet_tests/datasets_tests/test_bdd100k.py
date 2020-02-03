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

from lmnet.datasets.bdd100k import BDD100KObjectDetection, BDD100KSegmentation
from lmnet.datasets.dataset_iterator import DatasetIterator

# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


class DummyBDD100KObjDet(BDD100KObjectDetection):
    extend_dir = "BDD100K/bdd100k"


class DummyBDD100KSeg(BDD100KSegmentation):
    extend_dir = "BDD100K/seg"


def test_bdd100k_objdet():
    batch_size = 1
    train_dataset = DummyBDD100KObjDet(subset="train", batch_size=batch_size)
    train_iterator = DatasetIterator(train_dataset)

    test_dataset = DummyBDD100KObjDet(subset="validation", batch_size=batch_size)
    test_iterator = DatasetIterator(test_dataset)

    assert train_iterator.num_classes == 10
    colors = train_iterator.label_colors
    assert len(colors) == 10

    train_image_files, train_label_files = train_iterator.feed()
    assert train_image_files.shape[0] == batch_size
    assert train_label_files.shape[0] == batch_size

    test_image_files, test_label_files = test_iterator.feed()
    assert test_image_files.shape[0] == batch_size
    assert test_label_files.shape[0] == batch_size


def test_bdd100k_seg():
    batch_size = 1
    train_dataset = DummyBDD100KSeg(subset="train", batch_size=batch_size)
    train_iterator = DatasetIterator(train_dataset)

    test_dataset = DummyBDD100KSeg(subset="validation", batch_size=batch_size)
    test_iterator = DatasetIterator(test_dataset)

    assert train_dataset.num_classes == 41
    colors = train_dataset.label_colors
    assert len(colors) == 41

    train_image_files, train_label_files = train_iterator.feed()
    assert train_image_files.shape[0] == batch_size
    assert train_label_files.shape[0] == batch_size

    test_image_files, test_label_files = test_iterator.feed()
    assert test_image_files.shape[0] == batch_size
    assert test_label_files.shape[0] == batch_size
