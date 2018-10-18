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
"""TVM Runtime module used to improve CPU performance."""
import tvm
import utils
from os import path
from core.graph import Graph
from core.operators import Operator, Conv
from typing import Dict


class TVMLibraryGenerator(object):

    def __init__(self, graph: Graph, output_path: str, archs: Dict[str, str]) -> None:
        self.archs = archs
        self.graph = graph
        self.root_path = path.join(output_path, 'tvm_runtime', 'lib')

        for arch in archs:
            utils.make_dirs(path.join(self.root_path, arch))

    def __call__(self, node: Operator) -> None:
        func_name = 'tvm_gen_' + node.op_type
        getattr(self, func_name)(node)

    def tvm_gen_Conv(self, node: Conv) -> None:
        if node.is_quantized:
            return

        x_node = node.input_ops['X']
        w_node = node.input_ops['W']

        batch = 1
        kernel_height = node.kernel_height
        kernel_width = node.kernel_width
        in_channel = w_node.channel
        in_height = x_node.height
        in_width = x_node.width
        out_channel = node.channel
        out_height = node.height
        out_width = node.width
        pad_H = 0
        pad_W = 0
        stride_H = 1
        stride_W = 1
        if node.kernel_index_H and node.kernel_index_W:
            pad_H = node.pads[node.kernel_index_H] + \
                node.pads[node.kernel_index_H + node.kernel_dimensions]
            pad_W = node.pads[node.kernel_index_W] + \
                node.pads[node.kernel_index_W + node.kernel_dimensions]
            stride_H = node.strides[node.kernel_index_H]
            stride_W = node.strides[node.kernel_index_W]
        else:
            ValueError(f'Conv node {node.name} does not have kernel index for height and width.')

        A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
        W = tvm.placeholder((out_channel, kernel_height, kernel_width, in_channel), name='W')

        Apad = tvm.compute(
            (batch, in_height + pad_H, in_width + pad_W, in_channel),
            lambda nn, yy, xx, cc: tvm.select(
                tvm.all(yy >= pad_H, yy - pad_H < in_height,
                        xx >= pad_W, xx - pad_W < in_width),
                A[nn, yy - pad_H, xx - pad_W, cc], tvm.const(0.)),
            name='Apad')

        rc = tvm.reduce_axis((0, in_channel), name='rc')
        ry = tvm.reduce_axis((0, kernel_height), name='ry')
        rx = tvm.reduce_axis((0, kernel_width), name='rx')

        B = tvm.compute(
            (batch, out_height, out_width, out_channel),
            lambda nn, yy, xx, ff: tvm.sum(
                Apad[nn, yy * stride_H + ry, xx * stride_W + rx, rc] * W[ff, ry, rx, rc],
                axis=[ry, rx, rc]),
            name='B')
        s = tvm.create_schedule(B.op)

        libname = node.name
        filename = libname + '.o'
        for arch, target in self.archs.items():
            conv = tvm.build(s, [A, W, B], target=target, name=libname + '_tvm')
            conv.save(path.join(self.root_path, arch, filename))
