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
# Assuming the Python version is correct from here
import os
import shutil
import subprocess
import sys
from distutils.core import setup
from pathlib import Path

import pip
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

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
    'tensorflow==1.15.2',
    'click',
    'pyyaml',
    'jinja2',
    'pillow'
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


class CustomInstall(install):
    description = 'Install DLK'
    user_options = install.user_options

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)
        install.do_egg_install(self)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


setup(
    cmdclass={
        'install': CustomInstall},

    name='dlk',
    python_requires='>= 3.6.3',
    version=__version__,
    install_requires=install_requirements,
    extras_require={
        'validation': validation_requirements,
        'docbuild': docbuild_requirements,
    },
)
