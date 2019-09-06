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
import shutil
import unittest

from scripts import generate_project as gp


def tvm_is_available():
    available = True
    for module in ['tvm', 'topi']:
        try:
            __import__(module)
        except ImportError:
            available = False

    return available


class TestGenerateProject(unittest.TestCase):
    """Test class for 'generate_project.py' script."""

    @unittest.skipUnless(tvm_is_available(), "TVM is not available (reinstall with -tvm)")
    def test_generate_project_maximum_with_tvm(self) -> None:
        """Test code for testing 'generate_project.py' with maximum options"""
        output_path = os.path.join(os.getcwd(), 'tmp')
        input_path = os.path.abspath(
            os.path.join(os.getcwd(),
                         'examples',
                         'classification',
                         #  'lmnet_quantize_cifar10_stride_2.20180523.3x3',
                         'minimal_graph_with_shape.pb'))

        try:
            gp.run(input_path=input_path,
                   dest_dir_path=output_path,
                   project_name='unittest4',
                   activate_hard_quantization=True,
                   threshold_skipping=True,
                   num_pe=16,
                   use_tvm=True,
                   debug=False,
                   cache_dma=False,
                   )
        finally:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

        print("Script test with maximum options including TVM passed!")

    @unittest.skipUnless(tvm_is_available(), "TVM is not available (reinstall with -tvm)")
    def test_generate_project_with_no_quantize_with_tvm(self) -> None:
        """Test code for testing 'generate_project.py' with maximum options"""
        output_path = os.path.join(os.getcwd(), 'tmp')
        input_path = os.path.abspath(
            os.path.join(os.getcwd(),
                         'examples',
                         'classification',
                         # 'lmnet_quantize_cifar10_stride_2.20180523.3x3',
                         'minimal_graph_with_shape.pb'))

        try:
            gp.run(input_path=input_path,
                   dest_dir_path=output_path,
                   project_name='unittest5',
                   activate_hard_quantization=False,
                   threshold_skipping=True,
                   num_pe=16,
                   use_tvm=True,
                   debug=False,
                   cache_dma=False,
                   )
        finally:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

        print("Script test with no quantize but using TVM options passed!")

    @unittest.skipUnless(tvm_is_available(), "TVM is not available (reinstall with -tvm)")
    def test_generate_project_with_debug(self) -> None:
        """Test code for testing 'generate_project.py' with debug"""
        output_path = os.path.join(os.getcwd(), 'tmp')
        input_path = os.path.abspath(
            os.path.join(os.getcwd(),
                         'examples',
                         'classification',
                         'lmnet_quantize_cifar10',
                         'minimal_graph_with_shape.pb'))

        try:
            gp.run(input_path=input_path,
                   dest_dir_path=output_path,
                   project_name='unittest6',
                   activate_hard_quantization=True,
                   threshold_skipping=True,
                   num_pe=16,
                   use_tvm=True,
                   debug=True,
                   cache_dma=False,
                   )

        finally:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

        print("Script test with debug options passed!")


if __name__ == '__main__':
    unittest.main()
