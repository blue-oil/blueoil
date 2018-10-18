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
import utils


class Sequence(object):

    SKIP_PROCESSORS = ['ResizeWithGtBoxes']

    def __init__(self, processors=[]):
        self.processors = []

        for p in processors:
            for module_path, init_vals in p.items():
                name = self.module_basename(module_path)
                if name not in self.__class__.SKIP_PROCESSORS:
                    p_class = utils.dynamic_class_load(module_path)
                    p_inst = p_class(**init_vals)
                    self.processors.append(p_inst)
                else:
                    print(f'process {name} skipped because it\'s not implemented in DLK')

    def module_basename(self, module_path):
        return module_path.split('.')[-1]
