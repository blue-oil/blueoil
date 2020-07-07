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
import struct

from jinja2 import Environment as JinjaEnv
from jinja2 import FileSystemLoader
import numpy as np


def pack_to_bytes(a):
    t = type(a)
    if t == float or t == np.float32:
        packed_binary = struct.pack("<f", a)
    elif t == int or t == np.int32:
        packed_binary = struct.pack("<i", a)
    elif t == np.int64:
        packed_binary = struct.pack("<q", a)
    elif t == np.uint32:
        packed_binary = struct.pack("<I", a)

    if t == np.int64:
        return ",".join(map(str, list(struct.unpack("BBBBBBBB", packed_binary))))
    else:
        return ",".join(map(str, list(struct.unpack("BBBB", packed_binary))))


class Template(object):

    def __init__(self, config):
        self.jinja = self._create_jinja()
        self.config = config

    def generate(self, template_path, export_dir, new_name=None, **feed_dict):
        template_string = self.generate_string(template_path, **feed_dict)
        template_path = template_path.replace('.tpl', '')

        if new_name is None:
            template_name = path.basename(template_path)
        else:
            template_name = new_name

        export_path = path.join(export_dir, template_name)
        self._save_string_as_file(template_string, export_path)

        return export_path

    def manual_generate(self, template_path, export_dir, new_name=None, **feed_dict):
        template_path = path.join('manual', template_path)
        self.generate(template_path, export_dir, new_name, **feed_dict)

    def generate_string(self, name, **feed_dict):
        template = self.jinja.get_template(name)
        template_string = template.render(**self.config, **feed_dict)
        return template_string

    @property
    def root_dir(self):
        return path.join(path.dirname(path.abspath(__file__)), 'templates')

    def _save_string_as_file(self, string, file_path):
        with open(file_path, "w") as file:
            file.write(string)

    def _create_jinja(self):
        loader = FileSystemLoader(self.root_dir, encoding='utf8')
        jinja = JinjaEnv(loader=loader)
        jinja.globals['pack_to_bytes'] = pack_to_bytes

        return jinja
