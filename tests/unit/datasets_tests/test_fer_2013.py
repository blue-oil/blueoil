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

from blueoil.datasets.fer_2013 import FER2013

pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_fer2013():
    dataset = FER2013()
    assert len(dataset.classes) == 7
    assert dataset.available_subsets == ['train', 'test']
    assert len(dataset) == 1
    assert dataset[0][0].shape[0] == dataset.image_size
    assert dataset[0][0].shape[1] == dataset.image_size
    assert dataset[0][1] == 0
