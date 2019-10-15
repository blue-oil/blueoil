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
"""Test file for code generation for x86_64"""
from nose2.tools import params
from test_code_generation_base import TestCodeGenerationBase, get_configurations_by_architecture
from test_code_generation_base import get_configurations_by_test_cases, dict_codegen_classification
from tstutils import updated_dict, TEST_LEVEL_FUTURE_TARGET


def get_configurations_x86_64():
    cpu_name = "x86_64"
    test_cases = [
        {'use_avx': True, 'hard_quantize': True, 'threshold_skipping': True},
        {'use_avx': True, 'hard_quantize': True, 'threshold_skipping': False},
        {'use_avx': True, 'hard_quantize': False, 'threshold_skipping': False},
        {'use_avx': False, 'hard_quantize': True, 'threshold_skipping': True},
        {'use_avx': False, 'hard_quantize': True, 'threshold_skipping': False},
        {'use_avx': False, 'hard_quantize': False, 'threshold_skipping': False},
    ]
    configurations = get_configurations_by_architecture(test_cases, cpu_name)

    additional_test_configuration = updated_dict(dict_codegen_classification(cpu_name),
                                                 {'use_run_test_script': True})
    additional_test_cases = [
        {'use_avx': True, 'input_name': 'raw_image.png', 'test_level': TEST_LEVEL_FUTURE_TARGET},
        {'use_avx': True, 'input_name': 'preprocessed_image.npy', 'from_npy': True},
        {'use_avx': False, 'input_name': 'raw_image.png', 'test_level': TEST_LEVEL_FUTURE_TARGET},
        {'use_avx': False, 'input_name': 'preprocessed_image.npy', 'from_npy': True},
    ]
    configurations.extend(get_configurations_by_test_cases(additional_test_cases, additional_test_configuration))

    return [(i, configuration) for i, configuration in enumerate(configurations)]


class TestCodeGenerationX8664(TestCodeGenerationBase):
    """Test class for code generation testing for x86_64."""

    @params(*get_configurations_x86_64())
    def test_code_generation(self, i, configuration) -> None:
        self.run_test_all_configuration(i, configuration)
