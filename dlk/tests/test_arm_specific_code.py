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
import inspect
import os
from os.path import join
from pathlib import Path
import shutil
import unittest

from scripts import generate_project as gp
from testcase_dlk_base import TestCaseDLKBase
from tstutils import run_and_check


def get_func_name():
    return inspect.stack()[1][3]


def get_caller_func_name():
    return inspect.stack()[2][3]


class TestArmSpecific(TestCaseDLKBase):
    """Test for arm-specific code."""

    def test_arm_binary(self) -> None:
        """Test code for testing arm binary.

        This code assumes to be executed only on the ARM emulator environement."""

        cxx = 'arm-linux-gnueabihf-g++'
        cxx_path = shutil.which(cxx)
        if cxx_path is None:
            print('No arm compiler nor library. Quit testing.')
            raise unittest.SkipTest('No arm compiler nor library')
        else:
            arm_path = Path(cxx_path).parent.parent
        qemu = 'qemu-arm'
        # only works on Jenkins server
        arm_lib = arm_path.joinpath('arm-linux-gnueabihf').joinpath('libc')

        output_path = join(self.build_dir, get_func_name())
        model_path = os.path.join('examples', 'classification', 'lmnet_quantize_cifar10')
        input_dir_path = os.path.abspath(os.path.join(os.getcwd(), model_path))
        input_path = os.path.join(input_dir_path, 'minimal_graph_with_shape.pb')

        project_name = 'arm_binary'

        # code generation
        gp.run(input_path=str(input_path),
               dest_dir_path=str(output_path),
               project_name=str(project_name),
               activate_hard_quantization=True,
               threshold_skipping=False,
               num_pe=16,
               use_tvm=False,
               use_onnx=False,
               debug=False,
               cache_dma=False,
               )

        cpu_name = 'arm'
        bin_name = 'lm_' + cpu_name
        project_dir = Path(output_path).joinpath(project_name + '.prj')
        generated_bin = project_dir.joinpath(bin_name + '.elf')

        command0 = ['make', 'lm_arm', '-j8']

        run_and_check(command0,
                      project_dir,
                      join(output_path, "command0.out"),
                      join(output_path, "command0.err"),
                      self,
                      check_stdout_include=['g++'],
                      check_stdout_block=['error: ']
                      )

        self.assertTrue(os.path.exists(generated_bin))

        # prepare debug data
        debug_data_filename = 'cat.jpg'
        compressed_debug_data_path = os.path.join(input_dir_path, debug_data_filename + '.tar.gz')
        debug_data_path = os.path.join(output_path, debug_data_filename)
        debug_data_input = os.path.join(debug_data_path, '000_images_placeholder:0.npy')
        debug_data_output = os.path.join(debug_data_path, '133_output:0.npy')

        run_and_check(['tar', 'xvzf', str(compressed_debug_data_path), '-C', str(output_path)],
                      output_path,
                      join(output_path, "tar_xvzf.out"),
                      join(output_path, "tar_xvzf.err"),
                      self,
                      check_stdout_include=[debug_data_filename + '/raw_image.npy']
                      )

        self.assertTrue(os.path.exists(debug_data_input))
        self.assertTrue(os.path.exists(debug_data_output))

        command1 = [qemu, '-L', str(arm_lib)] if shutil.which(qemu) is not None else []
        command1 += [str(generated_bin), str(debug_data_input), str(debug_data_output)]

        print("Running ", command1)

        run_and_check(command1,
                      project_dir,
                      join(output_path, "command1.out"),
                      join(output_path, "command1.err"),
                      self,
                      check_stdout_include=['succeeded']
                      )

        print(f"arm binary test passed!")

    def test_arm_specific_code(self) -> None:
        """Test code for testing arm-specific code.

        This code assumes to be executed only on the ARM emulator environement."""

        cxx = 'arm-linux-gnueabihf-g++'
        qemu = 'qemu-arm'
        # only works on Jenkins server
        arm_lib = '/opt/x-tools/arm-unknown-linux-gnueabihf/arm-unknown-linux-gnueabihf/sysroot/'
        if shutil.which(cxx) is None or (shutil.which(qemu) is not None and not os.path.exists(arm_lib)):
            print('No arm compiler nor library. Quit testing.')
            raise unittest.SkipTest('No arm compiler nor library')

        output_path = join(self.build_dir, get_func_name())
        model_path = os.path.join('examples', 'classification', 'lmnet_quantize_cifar10')
        input_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), model_path)), 'minimal_graph_with_shape.pb')
        project_name = 'arm_specific'

        # code generation
        gp.run(input_path=input_path,
               dest_dir_path=output_path,
               project_name=project_name,
               activate_hard_quantization=True,
               threshold_skipping=False,
               num_pe=16,
               use_tvm=False,
               use_onnx=False,
               debug=False,
               cache_dma=False,
               )

        cpu_name = 'arm'
        lib_name = 'test_' + cpu_name
        project_dir = os.path.join(output_path, project_name + '.prj')
        generated_bin = os.path.join(project_dir, lib_name + '.elf')

        flags = ['-I./include',
                 '-std=c++0x',
                 '-O3',
                 '-D__USE_PNG__',
                 '-mcpu=cortex-a9',
                 '-mfpu=neon',
                 '-mthumb',
                 '-s',
                 '-static']

        cxxflags = ['-D__ARM__']

        # TODO it's better to fix makefile
        command0 = [cxx] + cxxflags + flags
        commands = [command0 + ['-c', 'src/pack2b_neonv7.S'],
                    command0 + ['-c', 'src/pack_input_to_qwords.cpp'],
                    command0 + ['-c', 'src/time_measurement.cpp'],
                    command0 + ['-c', 'mains/test_arm_main.cpp'],
                    command0 + ['pack2b_neonv7.o', 'test_arm_main.o', 'time_measurement.o', 'pack_input_to_qwords.o', '-lpthread', '-o',
                                generated_bin]
                    ]

        for i, command in enumerate(commands):
            run_and_check(command,
                          project_dir,
                          join(output_path, "command0-" + str(i) + ".out"),
                          join(output_path, "command0-" + str(i) + ".err"),
                          self
                          )

        self.assertTrue(os.path.exists(generated_bin))

        command1 = [qemu, '-L', arm_lib] if shutil.which(qemu) is not None else []
        command1 += [generated_bin]

        print("Running ", command1)

        run_and_check(command1,
                      project_dir,
                      join(output_path, "command1.out"),
                      join(output_path, "command1.err"),
                      self,
                      check_stdout_include=["Succeeded"]
                      )

        print(f"arm-specific code test passed!")


if __name__ == '__main__':
    unittest.main()
