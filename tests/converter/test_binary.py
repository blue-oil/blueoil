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
"""Test file for lm_xxxx binaries."""
import os
import unittest
from os.path import join

from blueoil.converter import generate_project as gp
from testcase_dlk_base import TestCaseDLKBase
from tstutils import run_and_check


class TestBinary(TestCaseDLKBase):
    """Test class for code generation testing."""

    def test_time_measurement_with_x86(self) -> None:
        """Test code for time measurement on x86."""
        model_path = os.path.join(
            'tests',
            'fixtures',
            'classification',
            'lmnet_quantize_cifar10')
        output_path = self.build_dir
        project_name = 'test_binary'
        project_dir = os.path.join(output_path, project_name + '.prj')
        generated_bin = os.path.join(project_dir, 'lm_x86_64')
        input_dir_path = os.path.abspath(os.path.join(os.getcwd(), model_path))
        input_path = os.path.join(input_dir_path, 'minimal_graph_with_shape.pb')
        debug_data_filename = 'cat.jpg'
        compressed_debug_data_path = os.path.join(input_dir_path, debug_data_filename + '.tar.gz')
        debug_data_path = os.path.join(output_path, debug_data_filename)
        debug_data_input = os.path.join(debug_data_path, '000_images_placeholder:0.npy')
        debug_data_output = os.path.join(debug_data_path, '133_output:0.npy')

        gp.run(input_path=input_path,
               dest_dir_path=output_path,
               project_name=project_name,
               activate_hard_quantization=False,
               threshold_skipping=False,
               debug=False,
               cache_dma=False,
        )
        self.assertTrue(os.path.exists(project_dir))

        run_and_check(['cmake', '.'],
                      project_dir,
                      join(project_dir, "make.out"),
                      join(project_dir, "make.err"),
                      self)

        run_and_check(['make', 'lm', '-j8'],
                      project_dir,
                      join(project_dir, "cmake.out"),
                      join(project_dir, "cmake.err"),
                      self)

        self.assertTrue(os.path.exists(generated_bin))

        run_and_check(['tar', 'xvzf', str(compressed_debug_data_path), '-C', str(output_path)],
                      output_path,
                      join(output_path, "tar_xvzf.out"),
                      join(output_path, "tar_xvzf.err"),
                      self,
                      check_stdout_include=[debug_data_filename + '/raw_image.npy']
        )

        self.assertTrue(os.path.exists(debug_data_input))
        self.assertTrue(os.path.exists(debug_data_output))

        run_and_check([str(generated_bin), str(debug_data_input), str(debug_data_output)],
                      project_dir,
                      join(project_dir, "elf.out"),
                      join(project_dir, "elf.err"),
                      self,
                      check_stdout_include=['TotalRunTime ']
        )

        print(f"Binary time-measurement test : passed!")


if __name__ == '__main__':
    unittest.main()
