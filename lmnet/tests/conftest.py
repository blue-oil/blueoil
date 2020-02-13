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
import tensorflow as tf

from blueoil import environment


@pytest.fixture
def reset_default_graph():
    """Reset tensorflow default graph."""
    print("reset_default_graph")
    tf.reset_default_graph()


@pytest.fixture
def set_test_environment():
    """Set test environment"""
    print("set test environment")

    yield environment.setup_test_environment()

    # By using a yield statement instead of return, all the code after the yield statement serves as the teardown code:
    # See also: https://docs.pytest.org/en/latest/fixture.html#fixture-finalization-executing-teardown-code
    environment.teardown_test_environment()
