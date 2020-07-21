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
import os
import json

from blueoil.cmd.profile_model import (
    run,
    _save_json
)
from blueoil.cmd.train import run as train_run
from blueoil.environment import setup_test_environment
from blueoil import environment

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_profile():

    config_file = "unit/fixtures/configs/for_profile.py"
    expriment_id = "test_profile"
    train_run(config_file, expriment_id, recreate=True, profile_step=7)

    setup_test_environment()

    run(expriment_id, None, None, 2, [])


def test_save_json():
    experiment_id = "test_save_json"
    environment.init(experiment_id)
    setup_test_environment()
    test_dict = {
        "model_name": "save_json",
        "image_size_height": 128,
        "image_size_width": 64,
        "num_classes": 3,
        "parameters": "test_node",
        "flops": "test_flops",
    }
    if not os.path.exists(environment.EXPERIMENT_DIR):
        os.makedirs(environment.EXPERIMENT_DIR)

    _save_json(
        name=test_dict["model_name"],
        image_size=(test_dict["image_size_height"], test_dict["image_size_width"]),
        num_classes=test_dict["num_classes"],
        node_param_dict=test_dict["parameters"],
        node_flops_dict=test_dict["flops"]
    )
    output_file = os.path.join(environment.EXPERIMENT_DIR, "{}_profile.json".format(test_dict["model_name"]))
    with open(output_file, 'r') as fp:
        file_data = json.load(fp)

    assert os.path.isfile(output_file)
    assert test_dict == file_data


if __name__ == '__main__':
    setup_test_environment()
    test_profile()
