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

from lmnet import environment
from lmnet.utils.config import _load_py, check_config, save_yaml

pytestmark = pytest.mark.usefixtures("set_test_environment")


def test_core_configs():
    """Test that all config files in `configs/core` dir include requirement keys."""
    dir_path = os.path.join("configs", "core")

    for config_file in glob.glob(os.path.join(dir_path, "**", "*.py"), recursive=True):
        config = _load_py(config_file)
        check_config(config, "training")
        check_config(config, "inference")


def test_convert_weight_from_darknet_configs():
    """Test that all config files in `configs/convert_weight_from_darknet` dir include requirement keys."""
    dir_path = os.path.join("configs", "convert_weight_from_darknet")

    for config_file in glob.glob(os.path.join(dir_path, "**", "*.py"), recursive=True):
        config = _load_py(config_file)
        check_config(config, "inference")


def test_example_config():
    """Test that example config python file include requirement keys."""

    dir_path = os.path.join("configs", "example")

    for config_file in glob.glob(os.path.join(dir_path, "**", "*.py"), recursive=True):
        config = _load_py(config_file)
        check_config(config, "training")
        check_config(config, "inference")


def test_example_classification_config_yaml():
    """Test that export config and meta yaml from example classification config python."""

    config_file = os.path.join("configs", "example", "classification.py")

    config = _load_py(config_file)

    config_yaml = os.path.join("configs", "example", "classification.yaml")

    config_meta = os.path.join("configs", "example", "classification_meta.yaml")

    environment.init("test_example_classification_config_yaml")
    saved_config, saved_meta = save_yaml(environment.EXPERIMENT_DIR, config)

    print(saved_meta)
    with open(config_yaml) as f:
        expected = f.read()
    with open(saved_config) as f:
        data = f.read()
        assert expected == data

    with open(config_meta) as f:
        expected = f.read()
    with open(saved_meta) as f:
        data = f.read()
        assert expected == data


def test_example_object_detection_config_yaml():
    """Test that export config and meta yaml from example object_detection config python."""

    config_file = os.path.join("configs", "example", "object_detection.py")

    config = _load_py(config_file)

    config_yaml = os.path.join("configs", "example", "object_detection.yaml")

    config_meta = os.path.join("configs", "example", "object_detection_meta.yaml")

    environment.init("test_example_object_detection_config_yaml")
    saved_config, saved_meta = save_yaml(environment.EXPERIMENT_DIR, config)

    with open(config_yaml) as f:
        expected = f.read()
    with open(saved_config) as f:
        data = f.read()
        assert expected == data

    with open(config_meta) as f:
        expected = f.read()
    with open(saved_meta) as f:
        data = f.read()
        assert expected == data


if __name__ == '__main__':
    test_core_configs()
    test_convert_weight_from_darknet_configs()
    test_example_config()
    test_example_classification_config_yaml()
    test_example_object_detection_config_yaml()
