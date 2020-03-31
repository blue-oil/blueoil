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
import copy
import glob
import os
import sys
import unittest


class DeviceE2eTest(unittest.TestCase):
    """Test Case of Device E2E Test."""
    sys_path_default = copy.deepcopy(sys.path)

    def _get_param(self, test_case, input_path, lib_name):
        test_case_dir = os.path.join(input_path, test_case)
        output_dir = glob.glob(os.path.join(test_case_dir, "export/*/*/output"))
        if output_dir:
            output_dir = output_dir[0]
        else:
            message = "No such directory: '{}'".format(os.path.join(test_case_dir, "export/*/*/output"))
            raise FileNotFoundError(message) if sys.version_info.major == 3 else IOError(message)
        test_data_dir = os.path.join(os.path.dirname(output_dir), "inference_test_data")
        model_dir = os.path.join(output_dir, "models")
        lib_dir = os.path.join(model_dir, "lib")
        return {
            "python_path": os.path.join(output_dir, "python"),
            'image': os.path.join(test_data_dir, "raw_image.png"),
            'model': os.path.join(lib_dir, lib_name),
            'config': os.path.join(model_dir, "meta.yaml"),
            }

    def _get_test_cases(self, input_path, lib_name):
        return [[case, self._get_param(case, input_path, lib_name)] for case in os.listdir(input_path)]

    def _run(self, python_path, image, model, config):
        sys.path = copy.deepcopy(self.sys_path_default) + [python_path]
        from run import run_prediction
        run_prediction(image, model, config)
        self.assertTrue(os.path.exists(os.path.join('output', "output.json")))

    def test_run(self):
        input_path = os.environ['DEVICE_TEST_INPUT_PATH']
        lib_name = os.environ['DEVICE_TEST_LIB_NAME']
        test_cases = self._get_test_cases(input_path, lib_name)
        for test_case_name, params in test_cases:
            print("Testing case: {}".format(test_case_name))
            if sys.version_info.major == 2:
                self._run(**params)
            else:
                with self.subTest(test_case_name=test_case_name, params=params):
                    self._run(**params)


if __name__ == "__main__":
    unittest.main()
