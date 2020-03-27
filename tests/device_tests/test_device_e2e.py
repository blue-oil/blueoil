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
import glob
import os
import pathlib
import sys
import unittest


class DeviceE2eTest(unittest.TestCase):
    """Base class for Device Test."""

    def get_param(self, test_case_path):
        output_dir = glob.glob(os.path.join(test_case_path, "export/*/*/output"))[0]
        test_data_dir = os.path.join(os.path.dirname(output_dir), "inference_test_data")
        model_dir = os.path.join(output_dir, "models")
        lib_dir = os.path.join(model_dir, "lib")
        return {
            "python_path": os.path.join(os.path.join(output_dir, "python"), "run,py"),
            'image': os.path.join(test_data_dir, "raw_image.png"),
            'model': os.path.join(lib_dir, self.lib_name),
            'config': os.path.join(model_dir, "meta.yaml"),
            }

    def get_test_cases(self):
        input_path = os.environ['DEVICE_TEST_INPUT_PATH']
        p = pathlib.Path(input_path)
        test_case_paths = [d for d in p.iterdir() if d.is_dir()]
        return [[path.name, self.get_param(path)] for path in test_case_paths]

    def run(self, python_path, image, model, config):
        sys.path.append(python_path)
        from run import run_prediction
        run_prediction(image, model, config)

    def test_run(self):
        test_cases = self.get_test_cases()
        self.lib_name = os.environ['DEVICE_TEST_LIB_NAME']

        for test_case_name, params in test_cases:
            with self.subTest(test_case_name=test_case_name, params=params):
                print("Testing case: {}".format(test_case_name))
                self.run(**params)
                assert os.path.exists(
                    os.path.join(os.path.join(params["python_path"], 'output'), "output.json")
                    )


if __name__ == "__main__":
    unittest.main()
