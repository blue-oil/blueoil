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

from executor.profile_model import run
from executor.train import run as train_run
from nn.environment import setup_test_environment

# Apply reset_default_graph() in conftest.py to all tests in this file.
# Set test environment
pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def test_profile():

    config_file = "tests/fixtures/configs/for_profile.py"
    expriment_id = "test_profile"
    train_run(None, None, config_file, expriment_id, recreate=True)

    setup_test_environment()

    run(expriment_id, None, None, 2, [])


if __name__ == '__main__':
    setup_test_environment()
    test_profile()
