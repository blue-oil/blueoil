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
from easydict import EasyDict

from lmnet.utils import config as config_util


def test_merge():

    base_config = EasyDict({"a": "aa", "nest": EasyDict({"b": "bb", "c": "cc"}), "d": "dd"})
    override_config = EasyDict({"a": "_a", "nest": EasyDict({"b": "_b"})})

    expected = EasyDict({"a": "_a", "nest": EasyDict({"b": "_b", "c": "cc"}), "d": "dd"})

    config = config_util.merge(base_config, override_config)
    assert config == expected


if __name__ == '__main__':
    test_merge()
