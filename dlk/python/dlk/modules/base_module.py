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
from os import path
import re
import ctypes as ct


class BaseModule(object):

    MODULE_DIR = path.abspath(path.dirname(__file__))
    ROOT_DIR = path.abspath(path.join(MODULE_DIR, '..', '..', '..'))
    BUILD_DIR = path.abspath(path.join(ROOT_DIR, 'build'))

    def __init__(self):
        self.lib = ct.cdll.LoadLibrary(self.lib_path)

    @property
    def lib_path(self):
        return path.join(self.build_dir, self.lib_name)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def lib_name(self):
        name = re.sub('(?<!^)(?=[A-Z])', '_', self.name).lower()
        return f'lib{name}.so'

    @property
    def build_dir(self):
        return self.__class__.BUILD_DIR
