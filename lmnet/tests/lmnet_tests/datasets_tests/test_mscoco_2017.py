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

from lmnet.pre_processor import ResizeWithJoints, JointsToGaussianHeatmap
from blueoil.datasets.mscoco_2017 import MscocoSinglePersonKeypoints
from blueoil.datasets.dataset_iterator import DatasetIterator
from lmnet.data_processor import Sequence


# Apply set_test_environment() in conftest.py to all tests in this file.
pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_mscoco_2017_single_pose_estimation():

    batch_size = 1
    image_size = [256, 320]
    stride = 2

    pre_processor = Sequence([ResizeWithJoints(image_size=image_size),
                              JointsToGaussianHeatmap(image_size=image_size,
                                                      stride=stride)])

    dataset = MscocoSinglePersonKeypoints(subset="train",
                                          batch_size=batch_size,
                                          pre_processor=pre_processor)
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == image_size[0] // stride
        assert labels.shape[2] == image_size[1] // stride
        assert labels.shape[3] == 17

    dataset = MscocoSinglePersonKeypoints(subset="validation",
                                          batch_size=batch_size,
                                          pre_processor=pre_processor)
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        images, labels = dataset.feed()

        assert isinstance(images, np.ndarray)
        assert images.shape[0] == batch_size
        assert images.shape[1] == image_size[0]
        assert images.shape[2] == image_size[1]
        assert images.shape[3] == 3

        assert isinstance(labels, np.ndarray)
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == image_size[0] // stride
        assert labels.shape[2] == image_size[1] // stride
        assert labels.shape[3] == 17
