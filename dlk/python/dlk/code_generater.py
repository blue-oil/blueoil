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
from pathlib import Path
import shutil

import utils
from template import Template

from core.config import Config
from core.graph import Graph
from core.operators import Conv
from typing import cast
from collections import defaultdict


class CodeGenerater(object):

    def __init__(self,
                 graph: Graph,
                 params,
                 config: Config) -> None:
        self.graph = graph
        self.params = params
        self.config = config
        assert len(self.graph.get_inputs()) == 1, 'Codegenerator does not support multiple inputs.'
        self.template = Template({
            'graph': self.graph,
            'params': self.params,
            'config': self.config,
            'graph_input': self.graph.get_inputs()[0],
            'graph_output': self.graph.non_variables[-1],
        })
        self.src_dir = path.join(self.config.output_pj_path, 'src')
        self.header_dir = path.join(self.config.output_pj_path, 'include')

    def generate_files_from_template(self) -> None:
        src_dir_path = self.template.root_dir
        file_pathes = utils.get_files(src_dir_path, excepts='/templates/manual')

        for src_file_path in file_pathes:
            src_file = Path(src_file_path)

            if src_file.is_file():
                relative_file_path = str(src_file.relative_to(src_dir_path))

                dest_file_path = path.join(self.config.output_pj_path, relative_file_path)
                dest_file_dir_path = path.dirname(dest_file_path)

                # if the file's dir not exist, make it
                utils.make_dirs([dest_file_dir_path])

                if 'tpl' in path.basename(src_file_path) and path.basename(src_file_path)[0] != '.':
                    relative_src_file_path = str(src_file.relative_to(self.template.root_dir))
                    self.template.generate(relative_src_file_path,
                                           dest_file_dir_path)
                else:
                    shutil.copy2(src_file_path, dest_file_path)

    def generate_consts(self) -> None:
        const_src_dir_path = path.join(self.src_dir, 'consts')
        const_header_dir_path = path.join(self.header_dir, 'consts')
        utils.make_dirs([const_src_dir_path, const_header_dir_path])

        const_src_template_path = path.join('manual', 'input', 'const.tpl.cpp')
        const_header_template_path = path.join('manual', 'input', 'const.tpl.h')

        for const in self.graph.consts:

            self.template.generate(const_src_template_path,
                                   const_src_dir_path,
                                   new_name=const.name,
                                   const=const)

            self.template.generate(const_header_template_path,
                                   const_header_dir_path,
                                   new_name=const.name,
                                   const=const)

    def generate_inputs(self) -> None:
        input_src_dir_path = path.join(self.src_dir, 'inputs')
        input_header_dir_path = path.join(self.header_dir, 'inputs')
        utils.make_dirs([input_src_dir_path, input_header_dir_path])

        input_src_template_path = path.join('consts', 'input.tpl.cpp')
        input_header_template_path = path.join('consts', 'input.tpl.h')

        for node in self.graph.consts:
            self.template.manual_generate(input_src_template_path,
                                          input_src_dir_path,
                                          new_name=node.name + '.cpp',
                                          node=node)

            self.template.manual_generate(input_header_template_path,
                                          input_header_dir_path,
                                          new_name=node.name + '.h',
                                          node=node)

    def generate_thresholds(self):
        src_template_path = path.join('manual', 'consts', 'thresholds.tpl.cpp')
        header_template_path = path.join('manual', 'consts', 'thresholds.tpl.h')

        qconvs_with_ts = [x for x in self.graph.non_variables
                          if x.op_type == 'Conv'
                          and cast(Conv, x).is_quantized
                          and cast(Conv, x).has_thresholds]

        self.template.generate(src_template_path,
                               self.src_dir,
                               quantized_convs=qconvs_with_ts)

        self.template.generate(header_template_path,
                               self.header_dir,
                               quantized_convs=qconvs_with_ts)

    def generate_scaling_factors(self):
        src_template_path = path.join('manual', 'consts', 'scaling_factors.tpl.cpp')
        header_template_path = path.join('manual', 'consts', 'scaling_factors.tpl.h')

        qconvs_convs = self.graph.convs(quantized_only=True)

        self.template.generate(src_template_path,
                               self.src_dir,
                               quantized_convs=qconvs_convs)

        self.template.generate(header_template_path,
                               self.header_dir,
                               quantized_convs=qconvs_convs)

    def generate_tvm_libraries(self):
        from core.tvm_library_generator import TVMLibraryGenerator
        supported_operations = ['Conv']

        archs = {
            'x86': 'llvm --system-lib -target=x86_64',
            'arm': 'llvm --system-lib -target=armv7l-none-linux-gnueabihf',
        }

        tvm_gen = TVMLibraryGenerator(self.graph, self.config.output_pj_path, archs)
        for operation in supported_operations:
            for node in self.graph.find_node_by_op_type(operation):
                tvm_gen(node)

    def reuse_output_buffers(self):

        operations = self.graph.non_variables
        candidates = defaultdict(set)

        for idx, op in enumerate(operations):
            prev_ops = operations[:idx]
            next_ops = operations[idx:]

            next_inputs = []
            for x in next_ops:
                for i in x.input_ops.values():
                    next_inputs.append(i.name)

            aliased = set()
            for prev_op in prev_ops:
                if prev_op.op_type in ['Reshape', 'Split']:
                    aliased.add(prev_op.name)
                    for i in prev_op.input_ops.values():
                        aliased.add(i.name)
                if prev_op.name not in next_inputs and prev_op.size >= op.size and prev_op.dtype == op.dtype:
                    candidates[op.name].add(prev_op.name)

            candidates[op.name] = candidates[op.name].difference(aliased)

        being_reused = []
        reusing = []
        for op in operations:
            cs = candidates[op.name]
            if cs:
                reusable_buffer = None
                for option in cs:
                    if option not in being_reused and option not in reusing:
                        reusable_buffer = option
                        break

                if reusable_buffer:
                    op.available_buffer = reusable_buffer
                    being_reused.append(reusable_buffer)
                    reusing.append(op.name)
