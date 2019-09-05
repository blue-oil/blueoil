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
import numpy as np
from lmnet.datasets.dataset_iterator import DatasetIterator
from lmnet.datasets.optical_flow_estimation import FlyingChairs, ChairsSDHom

pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_flying_chairs():
    batch_size = 1
    image_size = [384, 512]
    dataset = FlyingChairs(
        batch_size=batch_size)
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        image, flow = dataset.feed()
        assert isinstance(image, np.ndarray)
        assert image.shape[0] == batch_size
        assert image.shape[1] == image_size[0]
        assert image.shape[2] == image_size[1]
        assert image.shape[3] == 3 * 2

        assert isinstance(flow, np.ndarray)
        assert flow.shape[0] == batch_size
        assert flow.shape[1] == image_size[0]
        assert flow.shape[2] == image_size[1]
        assert flow.shape[3] == 2


def test_chairs_sdhom():
    batch_size = 1
    image_size = [384, 512]
    dataset = ChairsSDHom(
        batch_size=batch_size)
    dataset = DatasetIterator(dataset)

    for _ in range(5):
        image, flow = dataset.feed()

        assert isinstance(image, np.ndarray)
        assert image.shape[0] == batch_size
        assert image.shape[1] == image_size[0]
        assert image.shape[2] == image_size[1]
        assert image.shape[3] == 3 * 2

        assert isinstance(flow, np.ndarray)
        assert flow.shape[0] == batch_size
        assert flow.shape[1] == image_size[0]
        assert flow.shape[2] == image_size[1]
        assert flow.shape[3] == 2


if __name__ == '__main__':
    from lmnet.environment import setup_test_environment
    setup_test_environment()
    test_flying_chairs()
    test_chairs_sdhom()
