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

from nn.common import get_color_map

pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_get_color_map_with_small_length():
    color_map = get_color_map(5)
    assert len(color_map) == 5
    assert color_map[0] == (192, 0, 128)
    assert color_map[4] == (64, 0, 128)


def test_get_color_map_with_large_length():
    color_map = get_color_map(30)
    assert len(color_map) == 30
    assert color_map[0] == (192, 0, 128)
    assert color_map[29] == (128, 0, 192)
