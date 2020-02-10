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
import pytest

from blueoil.datasets.mscoco import MscocoObjectDetection


pytestmark = pytest.mark.usefixtures("set_test_environment")


# TODO(takecore): test segmentation


@pytest.mark.parametrize(
    "subset, num_classes, files, annotations", [
        ('train', 80, [
            'tests/fixtures/datasets/MSCOCO/train2014/COCO_train2014_000000000001.jpg',
            'tests/fixtures/datasets/MSCOCO/train2014/COCO_train2014_000000000002.jpg',
        ], [
            # [x, y, w, h, class_id]
            [[0, 0, 3, 3, 0], [7, 7, 3, 3, 0]], [[3, 3, 3, 3, 0], [6, 6, 3, 3, 0]]
        ]),
        ('validation', 80, [
            'tests/fixtures/datasets/MSCOCO/val2014/COCO_val2014_000000000001.jpg',
            'tests/fixtures/datasets/MSCOCO/val2014/COCO_val2014_000000000002.jpg',
        ], [
            [[0, 0, 3, 3, 0], [7, 7, 3, 3, 0]], [[3, 3, 3, 3, 0], [6, 6, 3, 3, 0]]
        ]),
    ]
)
def test_mscoco_object_detection(subset, num_classes, files, annotations):
    dataset = MscocoObjectDetection(subset=subset)

    assert dataset.num_classes == num_classes
    assert (files, annotations) == dataset._files_and_annotations()


def test_mscoco_object_detection_optional_classes_instantiate():
    '''Test override classes and instantiate
    '''
    MscocoObjectDetection.classes = ['person']
    MscocoObjectDetection()
