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
import pytest

from lmnet.common import Tasks
from lmnet.data_augmentor import SSDRandomCrop
from lmnet.utils.config import (
    merge,
    _check_config_augmentor,
)


def test_merge():

    base_config = EasyDict({"a": "aa", "nest": EasyDict({"b": "bb", "c": "cc"}), "d": "dd"})
    override_config = EasyDict({"a": "_a", "nest": EasyDict({"b": "_b"})})

    expected = EasyDict({"a": "_a", "nest": EasyDict({"b": "_b", "c": "cc"}), "d": "dd"})

    config = merge(base_config, override_config)
    assert config == expected


def test_check_config_augmentor():
    config = EasyDict({
        "TASK": Tasks.OBJECT_DETECTION,
        "DATASET": {
            "AUGMENTOR": [
                SSDRandomCrop(),
            ],
        }
    })
    _check_config_augmentor(config)

    raise_exception_config = EasyDict({
        "TASK": Tasks.CLASSIFICATION,
        "DATASET": {
            "AUGMENTOR": [
                SSDRandomCrop(),
            ],
        }
    })

    with pytest.raises(Exception) as excinfo:
        _check_config_augmentor(raise_exception_config)

    assert "The SSDRandomCrop augmentation can't be used in IMAGE.CLASSIFICATION task" in str(excinfo.value)


if __name__ == '__main__':
    test_merge()
    test_check_config_augmentor()
