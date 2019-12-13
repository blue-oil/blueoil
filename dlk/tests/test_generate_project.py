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
"""Test file for generate_project.py"""
import os
import unittest

from parameterized import parameterized

from scripts import generate_project as gp
from testcase_dlk_base import TestCaseDLKBase


def params_generate():
    flags_hq_thskip = [(False, False), (True, True), (False, True)]
    return [(i, flag_hq, flag_thskip) for i, (flag_hq, flag_thskip) in enumerate(flags_hq_thskip)]

class TestGenerateProject(TestCaseDLKBase):
    """Test class for 'generate_project.py' script."""

    @parameterized.expand(params_generate())
    def test_generate_project(self, i, flag_hq, flag_thskip) -> None:

        input_path = os.path.abspath(
            os.path.join(os.getcwd(),
                         'examples',
                         'classification',
                         #  'lmnet_quantize_cifar10_stride_2.20180523.3x3',
                         'minimal_graph_with_shape.pb'))

        output_path = os.path.join(self.build_dir, 'test_generate_project', str(i))
        gp.run(input_path=input_path,
               dest_dir_path=output_path,
               project_name='unittest',
               activate_hard_quantization=flag_hq,
               threshold_skipping=flag_thskip,
               num_pe=16,
               debug=False
               )

        print("Passed!")


if __name__ == '__main__':
    unittest.main()
