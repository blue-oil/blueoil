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

from blueoil.cmd.predict import run
from executor.train import run as train_run
from blueoil.environment import setup_test_environment

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_predict_classification():

    config_file = "tests/fixtures/configs/for_predict_classification.py"
    expriment_id = "test_predict_classification"
    train_run(None, None, config_file, expriment_id, recreate=True)

    setup_test_environment()

    run("tests/fixtures/sample_images", "outputs", expriment_id, None, None, save_images=True)


def test_predict_object_detection():

    config_file = "tests/fixtures/configs/for_predict_object_detection.py"
    expriment_id = "test_predict_object_detection"
    train_run(None, None, config_file, expriment_id, recreate=True)

    setup_test_environment()

    run("tests/fixtures/sample_images", "outputs", expriment_id, None, None, save_images=True)


# TODO(wakisaka): Do test semantic_segmentation. It need to dataset class for segmentation.

if __name__ == '__main__':
    setup_test_environment()
    test_predict_classification()

    setup_test_environment()
    test_predict_object_detection()
