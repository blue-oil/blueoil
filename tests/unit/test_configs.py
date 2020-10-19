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
import glob
import os

import pytest

from blueoil.utils.config import _load_py, check_config

pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_core_configs():
    """Test that all config files in `configs/core` dir include requirement keys."""
    dir_path = os.path.join("configs", "core")

    for config_file in glob.glob(os.path.join(dir_path, "**", "*.py"), recursive=True):
        config = _load_py(config_file)
        check_config(config, "training")
        check_config(config, "inference")


def test_example_config():
    """Test that example config python file include requirement keys."""

    dir_path = os.path.join("..", "blueoil", "configs", "example")

    for config_file in glob.glob(os.path.join(dir_path, "**", "*.py"), recursive=True):
        config = _load_py(config_file)
        check_config(config, "training")
        check_config(config, "inference")


if __name__ == '__main__':
    test_core_configs()
    test_example_config()
