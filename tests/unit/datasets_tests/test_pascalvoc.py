# Copyright 2020 The Blueoil Authors. All Rights Reserved.
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

from blueoil.datasets.pascalvoc_2007 import Pascalvoc2007
from blueoil.datasets.pascalvoc_2012 import Pascalvoc2012
from blueoil.datasets.pascalvoc_2007_2012 import Pascalvoc20072012
from blueoil.datasets.pascalvoc_custom import PascalVOCCustom


pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_pascalvoc_2007():
    dataset = Pascalvoc2007()
    assert len(dataset.classes) == 20
    assert dataset.available_subsets == ['train', 'validation', 'test', 'train_validation']
    assert dataset._files_and_annotations() == (
        ['unit/fixtures/datasets/PASCALVOC_2007/VOCdevkit/VOC2007/JPEGImages/000085.jpg'],
        [[[21, 69, 232, 257, 14], [337, 11, 162, 187, 14]]]
    )


def test_pascalvoc_2012():
    dataset = Pascalvoc2012()
    assert len(dataset.classes) == 20
    assert dataset.available_subsets == ['train', 'validation', 'train_validation']
    assert dataset._files_and_annotations() == (
        ['unit/fixtures/datasets/PASCALVOC_2012/VOCdevkit/VOC2012/JPEGImages/2012_002012.jpg'],
        [[[190, 67, 145, 307, 14]]]
    )


def test_pascalvoc_2007_2012():
    dataset = Pascalvoc20072012()
    assert len(dataset.classes) == 20
    assert dataset.available_subsets == ['train', 'validation', 'test']
    assert dataset.files == [
        'unit/fixtures/datasets/PASCALVOC_2007/VOCdevkit/VOC2007/JPEGImages/000085.jpg',
        'unit/fixtures/datasets/PASCALVOC_2012/VOCdevkit/VOC2012/JPEGImages/2012_002012.jpg'
    ]
    assert dataset.annotations == [[[21, 69, 232, 257, 14], [337, 11, 162, 187, 14]],
                                   [[190, 67, 145, 307, 14]]]


def test_pascalvoccustom():
    classes = ['person']
    available_subsets = ['train', 'validation', 'test', 'train_validation']
    extend_dir = 'PASCALVOC_2007/VOCdevkit/VOC2007'
    dataset = PascalVOCCustom(classes, available_subsets, extend_dir)
    assert dataset.classes == classes
    assert dataset.available_subsets == available_subsets
    assert dataset._files_and_annotations() == (
        ['unit/fixtures/datasets/PASCALVOC_2007/VOCdevkit/VOC2007/JPEGImages/000085.jpg'],
        [[[21, 69, 232, 257, 0], [337, 11, 162, 187, 0]]])

    classes = ['person']
    available_subsets = ['train', 'validation', 'train_validation']
    extend_dir = 'PASCALVOC_2012/VOCdevkit/VOC2012'
    dataset = PascalVOCCustom(classes, available_subsets, extend_dir)
    assert dataset.classes == classes
    assert dataset.available_subsets == available_subsets
    assert dataset._files_and_annotations() == (
        ['unit/fixtures/datasets/PASCALVOC_2012/VOCdevkit/VOC2012/JPEGImages/2012_002012.jpg'],
        [[[190, 67, 145, 307, 0]]])
