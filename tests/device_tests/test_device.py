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
"""Test file for binary running on devices"""
import os
import unittest

from nnlib import NNLib as NNLib

import numpy as np


class DeviceTest(unittest.TestCase):
    """Base class for Device Test."""

    def get_param(self, test_case_path):
        return {
            'library': os.path.join(test_case_path, "lib.so"),
            'input_npy': os.path.join(test_case_path, "input.npy"),
            'expected_npy': os.path.join(test_case_path, "expected.npy"),
            }

    def get_test_cases(self):
        input_path = os.environ['DEVICE_TEST_INPUT_PATH']
        test_case_paths = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]

        return [[test_case, self.get_param(os.path.join(input_path, test_case))] for test_case in test_case_paths]

    def run_library(self, library, input_npy, expected_npy):

        proc_input = np.load(input_npy)
        expected_output = np.load(expected_npy)
        # load and initialize the generated shared library
        nn = NNLib()
        nn.load(library)
        nn.init()

        # run the graph
        batched_proc_input = np.expand_dims(proc_input, axis=0)
        output = nn.run(batched_proc_input)

        rtol = atol = 0.0001
        n_failed = expected_output.size - np.count_nonzero(np.isclose(output, expected_output, rtol=rtol, atol=atol))

        return n_failed == 0

    def test_run_library(self):
        test_cases = self.get_test_cases()

        for test_case_name, params in test_cases:
            with self.subTest(test_case_name=test_case_name, params=params):
                print(f"Testing case: {test_case_name}")
                self.assertTrue(self.run_library(**params))


if __name__ == "__main__":
    unittest.main()
