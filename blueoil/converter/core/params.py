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
"""Parameter module."""
from typing import List

from blueoil.converter.core.config import Config
from blueoil.converter.core.data_types import Uint32
from blueoil.converter.core.operators import Conv


class Params(object):
    """Parameter class."""

    from blueoil.converter.core.graph import Graph

    def __init__(self, graph: Graph, config: Config) -> None:
        """Init this parameter object.

        Parameters
        ----------
        graph : Graph
            Graph object

        config : Config
            Configuration object
        """

        self.graph = graph
        self.config = config

    @property
    def default_qword_dtype(self):
        return self.config.default_qword_dtype

    @property
    def default_nbit_qword(self):
        if self.default_qword_dtype == Uint32:
            return 32
        else:
            raise NotImplemented

    @property
    def nbit_qinput(self):
        return 2  # self.config.nbit_qinput

    @property
    def nbit_qkernel(self):
        return 1  # self.config.nbit_qkernel

    @property
    def max_nbit_qinput(self):
        return self.nbit_qinput

    @property
    def max_nbit_qkernel(self):
        return self.nbit_qkernel

    @property
    def num_qinputs_in_qword(self):
        return int(self.default_nbit_qword / self.nbit_qinput)

    @property
    def num_qkernels_in_qword(self):
        return int(self.default_nbit_qword / self.nbit_qkernel)

    @property
    def max_size_inputs_per_layer(self):
        node_max = max([x.size for x in self.graph.non_variables])
        assert len(self.graph.get_inputs()) == 1, \
            f"Currently, only one input is assumed {list(map(lambda x: x.name, self.graph.get_inputs()))}."
        return int(max([node_max, self.graph.get_inputs()[0].size]))

    @property
    def max_size_kn2row_col_block(self) -> int:
        return 256

    @property
    def max_size_kn2row_buffer_per_layer(self) -> int:
        convs: List[Conv] = self.graph.convs()

        kn2row_buffer_sizes = \
            [(
                x.kernel_height *
                x.kernel_width *
                min(self.max_size_kn2row_col_block, x.height * x.width) *
                x.channel
            ) for x in convs]

        return max(kn2row_buffer_sizes) if kn2row_buffer_sizes else 0

    @property
    def max_size_outputs_per_layer(self):
        node_max = max([x.size for x in self.graph.non_variables + self.graph.get_outputs()])
        return int(node_max)

    @property
    def max_size_kernels_per_layer(self) -> int:
        kernel_sizes = [x.size for x in self.graph.consts]
        assert kernel_sizes, "No kernels found."
        return int(max(kernel_sizes))

    @property
    def max_elems_kernel(self) -> int:
        kernel_elems = [x.height * x.width * x.channel for x in self.graph.consts if x.rank == 4]
        assert kernel_elems, "No kernels found."
        return int(max(kernel_elems))

    @property
    def max_size_qinputs_per_layer(self):
        # this is temporary because not every consts is kernel
        # also later, each layer has different bitwidth
        return int(self.max_size_inputs_per_layer / self.num_qinputs_in_qword)

    @property
    def max_size_qoutputs_per_layer(self):
        # this is temporary because not every consts is kernel
        # also later, each layer has different bitwidth
        return int(self.max_size_outputs_per_layer / self.num_qinputs_in_qword)

    @property
    def max_size_qkernels_per_layer(self):
        # this is temporary because not every consts is kernel
        return int(self.max_size_kernels_per_layer / self.num_qkernels_in_qword)

    @property
    def max_size_qkernels_per_pe(self):
        return int(self.max_elems_kernel)
