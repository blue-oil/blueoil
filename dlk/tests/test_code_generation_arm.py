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
"""Test file for code generation for arm"""
from nose2.tools import params
from test_code_generation_base import TestCodeGenerationBase, get_configurations_by_architecture


def get_configurations_arm():
    cpu_name = "arm"
    test_cases = [
        {'need_arm_compiler': True, 'hard_quantize': True, 'threshold_skipping': True},
        {'need_arm_compiler': True, 'hard_quantize': True, 'threshold_skipping': False},
        # FIXME: Cannot run library with hard_quantize=False for segmentation task.
        # Issue link: https://github.com/blue-oil/blueoil/issues/512
        # {'need_arm_compiler': True, 'hard_quantize': False, 'threshold_skipping': True},
        # {'need_arm_compiler': True, 'hard_quantize': False, 'threshold_skipping': False},
    ]
    configurations = get_configurations_by_architecture(test_cases, cpu_name)

    return [(i, configuration) for i, configuration in enumerate(configurations)]


class TestCodeGenerationArm(TestCodeGenerationBase):
    """Test class for code generation testing for arm."""

    @params(*get_configurations_arm())
    def test_code_generation(self, i, configuration) -> None:
        self.run_test_all_configuration(i, configuration)
