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
# check Python version
import sys
# Assuming the Python version is correct from here
import os
import pip
import subprocess
import shutil
from pathlib import Path
from setuptools import Extension
from setuptools.command.test import test
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

from distutils.core import setup
from python.dlk import __version__

CURRENT_PYTHON = sys.version_info[:3]
REQUIRED_PYTHON = (3, 6, 3)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """
        This version of DLK requires Python {}.{}.{},
        but you're trying to install it on Python {}.{}.{}.
        """.format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)


install_requirements = [
    'numpy',
    'tensorflow==1.13.1',
    'click',
    'pyyaml',
    'jinja2',
    'pillow'
]


tests_requirements = [
    'nose2',
]


validation_requirements = [
    'pycodestyle',
    'autopep8',
]

docbuild_requirements = [
    'sphinx',
    'recommonmark==0.4.0',
    'sphinx_rtd_theme',
]


def run_command(command):
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True)
    while proc.poll() is None:
        sys.stdout.write(proc.stdout.readline())
        sys.stdout.flush()
    ret_value = proc.wait()
    output = proc.communicate()

    if ret_value != 0:
        if output[1]:
            print(output[1])
        raise RuntimeError('Error executing command: ' + command)


def build_library():
    root_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(root_path)
    os.makedirs(root_path + '/build', exist_ok=True)
    os.chdir(root_path + '/build')
    os.system('cmake ../cpp/packer')
    os.system('make')
    os.chdir(root_path)


class CustomInstall(install):
    description = 'Install DLK'
    user_options = install.user_options
    user_options.append(('enable-tvm', None, 'Enable TVM support'))
    user_options.append(('enable-onnx', None, 'Enable ONNX support'))

    def initialize_options(self):
        install.initialize_options(self)
        self.enable_tvm = None
        self.enable_onnx = None

    def finalize_options(self):
        install.finalize_options(self)
        print('TVM support: ' + ('OFF' if self.enable_tvm is None else 'ON'))
        print('ONNX support: ' + ('OFF' if self.enable_onnx is None else 'ON'))

    def run(self):
        install.run(self)
        install.do_egg_install(self)

        if self.enable_tvm:
            # clone TVM from github
            if not os.path.exists('tvm'):
                run_command('git clone --recursive https://github.com/dmlc/tvm.git')

            # install TVM
            root_path = Path(os.path.dirname(os.path.realpath(__file__)))
            tvm_path = root_path / 'tvm'
            build_path = tvm_path / 'build'
            tvm_python_path = tvm_path / 'python'
            tvm_topi_path = tvm_path / 'topi' / 'python'
            tvm_runtime_code = tvm_path / 'apps' / 'howto_deploy' / 'tvm_runtime_pack.cc'
            tvm_runtime_code_dest_path = root_path / 'python' / 'dlk' / 'templates' / 'tvm_runtime'

            os.chdir(str(root_path))

            if tvm_runtime_code.exists():
                if not tvm_runtime_code_dest_path.exists():
                    tvm_runtime_code_dest_path.mkdir(parents=True)
                shutil.copy2(tvm_runtime_code, tvm_runtime_code_dest_path)

            already_installed = True
            for module in ['tvm', 'topi']:
                try:
                    __import__(module)
                except ImportError:
                    print('Module %s is not installed' % module)
                    already_installed = False

            if already_installed:
                print('TVM is already installed')
                return

            os.chdir(str(tvm_path))

            llvm_config_path = os.getenv('LLVM_CONFIG_PATH', False) or 'llvm-config'
            cpu_count = len(os.sched_getaffinity(0)) if 'sched_getaffinity' in dir(os) else os.cpu_count()
            cpu_count = min(cpu_count, 8)

            run_command('cp make/config.mk ./')
            run_command(f'echo "LLVM_CONFIG = {llvm_config_path}" >> ./config.mk')
            run_command(f'make -j{cpu_count}')

            os.environ['PYTHONPATH'] = str(tvm_python_path) + ':' + str(tvm_topi_path)

            os.chdir(str(tvm_python_path))
            run_command('python setup.py install')

            os.chdir(str(tvm_topi_path))
            run_command('python setup.py install')

            os.chdir(root_path)

        if self.enable_onnx:
            # install ONNX v1.1.1
            if int(pip.__version__.split('.')[0]) >= 10:
                from pip._internal import main as pipmain
                pipmain(['install', 'onnx==1.1.1'])
            else:
                from pip import main as pipmain
                pipmain(['install', 'onnx==1.1.1'])


class CustomTest(test):
    def run(self):
        build_library()
        test.run(self)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        self.build_extension()

    def build_extension(self):
        build_library()


setup(
    cmdclass={
        'test': CustomTest,
        'build_ext': CMakeBuild,
        'install': CustomInstall},

    ext_modules=[CMakeExtension('packer')],
    name='dlk',
    python_requires='>= 3.6.3',
    version=__version__,
    install_requires=install_requirements,
    tests_require=tests_requirements,
    test_suite='nose2.collector.collector',
    extras_require={
        'validation': validation_requirements,
        'docbuild': docbuild_requirements,
    },
)
