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
"""Test file for code generation"""
import numpy as np
import os
import inspect
from os.path import join, basename
import shutil
import sys
import unittest

from parameterized import parameterized

from scripts import generate_project as gp
from scripts.pylib.nnlib import NNLib as NNLib

sys.path.append("utils")  # PEP8:ignore
import run_test as inference  # PEP8:ignore

from testcase_dlk_base import TestCaseDLKBase
from tstconf import CURRENT_TEST_LEVEL
from tstutils import updated_dict, run_and_check, TEST_LEVEL_FUTURE_TARGET, FPGA_HOST


def dict_codegen_classification(cpu_name) -> dict:
    """Test parameters for testing code generation for classification on CPU """
    return {'model_path': os.path.join('examples', 'classification', 'lmnet_quantize_cifar10_space_to_depth'),
            'expected_output_set_name': '1000_dog.png',
            'prefix': 'cls',
            'input_name': '000_images_placeholder:0.npy',
            'output_npy_name': '133_output:0.npy',
            'cpu_name': cpu_name,
            'hard_quantize': True,
            'threshold_skipping': False,
            'use_avx': False
            }


def dict_codegen_classification_resnet(cpu_name) -> dict:
    """Test parameters for testing code generation for classification on CPU (float only)"""
    return {'model_path': os.path.join('examples', 'classification', 'resnet_quantize_cifar10'),
            'expected_output_set_name': '9984_horse.png',
            'prefix': 'cls_resnet',
            'input_name': '000_images_placeholder:0.npy',
            'output_npy_name': '368_output:0.npy',
            'cpu_name': cpu_name,
            'hard_quantize': True,
            'threshold_skipping': False,
            'use_avx': False
            }


def dict_codegen_object_detection(cpu_name) -> dict:
    """Test parameters for testing code generation for object detection on CPU"""
    return {'model_path': os.path.join('examples', 'object_detection', 'fyolo_quantize_4_v4'),
            'expected_output_set_name': 'network_input_output',
            'prefix': 'det',
            'input_name': '000_images_placeholder:0.npy',
            'output_npy_name': '317_output:0.npy',
            'cpu_name': cpu_name,
            'hard_quantize': True,
            'threshold_skipping': False,
            'use_avx': False
            }


def dict_codegen_segmentation(cpu_name) -> dict:
    """Test parameters for testing code generation for segmentation on CPU"""
    return {'model_path': os.path.join('examples', 'segmentation', 'lm_segnet_v1_quantize_camvid'),
            'expected_output_set_name': 'network_input_output',
            'prefix': 'seg',
            'input_name': '000_images_placeholder:0.npy',
            'output_npy_name': '227_output:0.npy',
            'cpu_name': cpu_name,
            'hard_quantize': True,
            'threshold_skipping': False,
            'use_avx': False
            }


def get_configurations_by_test_cases(test_cases, configuration):

    return [updated_dict(configuration,test_case) for test_case in test_cases]


def get_configurations_by_architecture(test_cases, cpu_name):
    configurations = []
    configurations.extend(get_configurations_by_test_cases(test_cases, dict_codegen_classification(cpu_name)))
    configurations.extend(get_configurations_by_test_cases(test_cases, dict_codegen_classification_resnet(cpu_name)))
    configurations.extend(get_configurations_by_test_cases(test_cases, dict_codegen_object_detection(cpu_name)))
    configurations.extend(get_configurations_by_test_cases(test_cases, dict_codegen_segmentation(cpu_name)))

    return configurations


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


def get_configurations_arm():
    cpu_name = "arm"
    test_cases = [
        {'need_arm_compiler': True, 'hard_quantize': True, 'threshold_skipping': True},
        {'need_arm_compiler': True, 'hard_quantize': True, 'threshold_skipping': False},
        {'need_arm_compiler': True, 'hard_quantize': False, 'threshold_skipping': True},
        {'need_arm_compiler': True, 'hard_quantize': False, 'threshold_skipping': False},
    ]
    configurations = get_configurations_by_architecture(test_cases, cpu_name)

    return [(i, configuration) for i, configuration in enumerate(configurations)]


def get_configurations_arm_fpga():
    cpu_name = "arm_fpga"
    test_cases = [
        {'need_arm_compiler': True, 'cache_dma': True, 'threshold_skipping': True},
        {'need_arm_compiler': True, 'cache_dma': True, 'threshold_skipping': False},
        {'need_arm_compiler': True, 'cache_dma': False, 'threshold_skipping': True},
        {'need_arm_compiler': True, 'cache_dma': False, 'threshold_skipping': False},
    ]
    configurations = get_configurations_by_architecture(test_cases, cpu_name)

    return [(i, configuration) for i, configuration in enumerate(configurations)]


def get_configurations_aarch64():
    cpu_name = "aarch64"
    test_cases = [
        {'hard_quantize': True, 'threshold_skipping': True},
        {'hard_quantize': True, 'threshold_skipping': False},
        {'hard_quantize': False, 'threshold_skipping': True},
        {'hard_quantize': False, 'threshold_skipping': False},
    ]
    configurations = get_configurations_by_architecture(test_cases, cpu_name)

    return [(i, configuration) for i, configuration in enumerate(configurations)]


class TestCodeGenerationBase(TestCaseDLKBase):
    """Test class for code generation testing."""

    def run_test_all_configuration(self, i, configuration) -> None:

        print(f"\nCode generation test: ID: {i}, Testcase: {configuration}")
        #  TODO consider better implementation
        this_test_level = configuration.get("test_level", 0)
        if this_test_level < CURRENT_TEST_LEVEL:
            self.codegen_cpu(test_id=i, **configuration)
        else:
            raise unittest.SkipTest(
                f'test level of this test: {this_test_level}, current test level: {CURRENT_TEST_LEVEL}')

    def save_output(self, test_id, library, input_npy, expected_output_npy):
        save_dir = os.path.join(self.class_output_dir, str(test_id))
        os.makedirs(save_dir, exist_ok=True)
        shutil.copyfile(library, os.path.join(save_dir, 'lib.so'))
        shutil.copyfile(input_npy, os.path.join(save_dir, 'input.npy'))
        shutil.copyfile(expected_output_npy, os.path.join(save_dir, 'expected.npy'))
        # change mode for docker use case with root user
        os.chmod(save_dir, 0o777)
        for f in os.listdir(save_dir):
            os.chmod(os.path.join(save_dir, f), 0o777)

    def run_library(self, library, input_npy, expected_output_npy):

        proc_input = np.load(input_npy)
        expected_output = np.load(expected_output_npy)
        # load and initialize the generated shared library
        nn = NNLib()
        nn.load(library)
        nn.init()

        # run the graph
        batched_proc_input = np.expand_dims(proc_input, axis=0)
        output = nn.run(batched_proc_input)

        rtol = atol = 0.0001
        n_failed = expected_output.size - np.count_nonzero(np.isclose(output, expected_output, rtol=rtol, atol=atol))
        percent_failed = (n_failed / expected_output.size) * 100.0

        return percent_failed

    def run_library_using_script(self, library: str, image: str, expected_output_npy: str, from_npy: bool) -> float:

        percent_failed = inference.main_test(image, library, expected_output_npy, from_npy=from_npy)
        return percent_failed

    def run_library_on_remote(self,
                              host: str,
                              output_path: str,
                              library: str,
                              input_npy: str, expected_output_npy: str) -> float:

        run_and_check(
            [ "ssh",
             f"root@{host}",
             f"rm -rf ~/automated_testing/*"
            ],
            output_path,
            join(output_path, "clean.err"),
            join(output_path, "clean.err"),
            self)

        lib_name = os.path.basename(library)
        input_name = os.path.basename(input_npy)
        output_name = os.path.basename(expected_output_npy)

        run_library_code  =  "import numpy as np\n"
        run_library_code +=  "from nnlib import NNLib as NNLib\n"
        run_library_code +=  "class testing:\n"
        run_library_code +=  inspect.getsource(self.run_library)
        run_library_code +=  "if __name__ == '__main__':\n"
        run_library_code +=  "  t = testing()\n"
        run_library_code += f"  print(t.run_library('./{lib_name}', './{input_name}', './{output_name}'))\n"

        testing_code_name = "testing_code.py"
        testing_code_path = join(output_path, testing_code_name)
        with open(testing_code_path, "w") as code_file:
            code_file.write(run_library_code)

        run_and_check(
            [ "scp",
              library,
              input_npy,
              expected_output_npy,
              inspect.getfile(NNLib),
              testing_code_path,
             f"root@{host}:~/automated_testing/"
            ],
            output_path,
            join(output_path, "scp.out"),
            join(output_path, "scp.err"),
            self)

        remote_output_file = join(output_path, "remote.out")
        run_and_check(
            [ "ssh",
             f"root@{host}",
             f"cd ~/automated_testing/; python {testing_code_name}"
            ],
            output_path,
            remote_output_file,
            join(output_path, "remote.err"),
            self,
            keep_outputs=True)

        with open(remote_output_file, "r") as remote_output_file:
            remote_output = remote_output_file.read()

        pf = 100.0
        try:
            pf = float(remote_output)
        except:
            pf = 100.0

        return pf

    def codegen_cpu(self,
                    model_path,
                    expected_output_set_name,
                    prefix,
                    input_name,
                    output_npy_name,
                    cpu_name='x86_64',
                    hard_quantize=False,
                    threshold_skipping=False,
                    use_run_test_script=False,
                    max_percent_incorrect_values=0.1,
                    from_npy=False,
                    need_arm_compiler=False,
                    cache_dma=False,
                    use_avx=False,
                    test_id=0
                    ) -> None:

        """Test code for testing code generation for CPU"""
        # TODO consider better implementation
        if need_arm_compiler:
            if shutil.which('arm-linux-gnueabihf-g++') is None:
                raise unittest.SkipTest('No arm compiler.')

        dir_tags = [str(test_id), prefix, basename(model_path), cpu_name]
        dir_tags = dir_tags + ['hq'] if hard_quantize else dir_tags
        dir_tags = dir_tags + ['thskip'] if threshold_skipping else dir_tags

        output_path = join(self.build_dir, '_'.join(dir_tags))
        input_dir_path = os.path.abspath(
            os.path.join(os.getcwd(),
                         model_path))

        input_path = os.path.join(input_dir_path, 'minimal_graph_with_shape.pb')
        project_name = 'code_generation'

        gp.run(input_path=input_path,
               dest_dir_path=output_path,
               project_name=project_name,
               activate_hard_quantization=hard_quantize,
               threshold_skipping=threshold_skipping,
               debug=False,
               cache_dma=cache_dma,
               )

        lib_name = 'libdlk_' + cpu_name
        project_dir = os.path.join(output_path, project_name + '.prj')
        generated_lib = os.path.join(project_dir, lib_name + '.so')
        npy_targz = os.path.join(input_dir_path, expected_output_set_name + '.tar.gz')

        run_and_check(['tar', 'xvzf', str(npy_targz), '-C', str(output_path)],
                      input_dir_path,
                      join(output_path, "tar_xvzf.out"),
                      join(output_path, "tar_xvzf.err"),
                      self,
                      check_stdout_include=[expected_output_set_name + '/raw_image.npy']
                      )

        npy_path = os.path.join(output_path, expected_output_set_name)
        input_path = os.path.join(npy_path, input_name)
        expected_output_path = os.path.join(npy_path, output_npy_name)

        self.assertTrue(os.path.exists(project_dir))

        cmake_use_aarch64 = '-DTOOLCHAIN_NAME=linux_aarch64'
        cmake_use_arm = '-DTOOLCHAIN_NAME=linux_arm'
        cmake_use_neon = '-DUSE_NEON=1'
        cmake_use_fpga = '-DRUN_ON_FPGA=1'
        cmake_use_avx = '-DUSE_AVX=1'

        cmake_defs = []
        if cpu_name == 'aarch64':
            cmake_defs += [cmake_use_aarch64, cmake_use_neon]
        elif cpu_name == 'arm':
            cmake_defs += [cmake_use_arm, cmake_use_neon]
        elif cpu_name == 'arm_fpga':
            cmake_defs += [cmake_use_arm, cmake_use_neon, cmake_use_fpga]
        elif cpu_name == 'x86_64':
            if use_avx:
                cmake_defs += [cmake_use_avx]

        run_and_check(['cmake'] + cmake_defs + ['.'],
                      project_dir,
                      join(output_path, "cmake.out"),
                      join(output_path, "cmake.err"),
                      self,
                      check_stdout_include=['Generating done'],
                      check_stdout_block=['CMake Error']
                      )

        run_and_check(['make', 'VERBOSE=1', 'lib', '-j8'],
                      project_dir,
                      join(output_path, "make.out"),
                      join(output_path, "make.err"),
                      self,
                      check_stdout_include=['Building'],
                      check_stderr_block=['error: ']
                      )
        self.assertTrue(os.path.exists(generated_lib))

        if not use_run_test_script:
            if cpu_name == 'x86_64':
                percent_failed = self.run_library(generated_lib, input_path, expected_output_path)
            elif cpu_name == 'arm' or cpu_name == 'arm_fpga':
                percent_failed = \
                    self.run_library_on_remote(FPGA_HOST, output_path, generated_lib, input_path, expected_output_path)
            elif cpu_name == 'aarch64':
                # Skip run library test, it will run on another test by using saved output files.
                self.save_output(test_id, generated_lib, input_path, expected_output_path)
                return
            else:
                self.fail("Unexpected cpu_name: %s" % cpu_name)
        else:
            percent_failed = self.run_library_using_script(generated_lib, input_path, expected_output_path,
                                                           from_npy)

        self.assertTrue(percent_failed < max_percent_incorrect_values,
                        msg=f"Test failed: {percent_failed:.3f}% of the values does not match")

        print(f"Codegen test {prefix}: passed!  {100.0 - percent_failed:.3f}% "
              f"of the output values are correct\n"
              f"[hard quantize == {hard_quantize}, threshold skipping == {threshold_skipping}, cache == {cache_dma}]")


class TestCodeGenerationX8664(TestCodeGenerationBase):
    """Test class for code generation testing."""

    @parameterized.expand(get_configurations_x86_64())
    def test_code_generation(self, i, configuration) -> None:
        self.run_test_all_configuration(i, configuration)


class TestCodeGenerationArm(TestCodeGenerationBase):
    """Test class for code generation testing."""

    fpga_setup = True

    @parameterized.expand(get_configurations_arm())
    def test_code_generation(self, i, configuration) -> None:
        self.run_test_all_configuration(i, configuration)


class TestCodeGenerationArmFpga(TestCodeGenerationBase):
    """Test class for code generation testing."""

    fpga_setup = True

    @parameterized.expand(get_configurations_arm_fpga())
    def test_code_generation(self, i, configuration) -> None:
        self.run_test_all_configuration(i, configuration)


class TestCodeGenerationAarch64(TestCodeGenerationBase):
    """Test class for code generation testing for aarch64."""

    @parameterized.expand(get_configurations_aarch64())
    def test_code_generation(self, i, configuration) -> None:
        self.run_test_all_configuration(i, configuration)


if __name__ == '__main__':
    unittest.main()
