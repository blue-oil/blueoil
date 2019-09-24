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
"""Test file for Graph."""
import unittest

import numpy as np

from core.data_types import Float32
from core.graph import Graph
from core.operators import Constant, Conv, Input, Output


class TestGraph(unittest.TestCase):
    """Test class for Graph."""

    def test_graph_conv(self) -> None:
        """Test code for making a simple graph with Conv."""
        graph = Graph()

        # two inputs
        x = Input(
            'input',
            [1, 5, 5, 3],
            Float32(),
        )

        w = Constant(
            'weight',
            Float32(),
            np.zeros([1, 2, 2, 3])
        )

        # Conv
        conv = Conv(
            'conv',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': w},  # you can get these keys by 'Conv.input_names'
            kernel_shape=[2, 2]
        )

        # One output
        y = Output(
            'output',
            [1, 4, 4, 3],
            Float32(),
            {'input': conv}  # you can get this key by 'Output.input_names'
        )

        # add ops to the graph
        graph.add_op(x)
        graph.add_op(w)
        graph.add_op(conv)
        graph.add_op(y)

        self.assertTrue(graph.check_nodes(), "All inputs of operators must match their outputs.")
        print("Graph test passed!")


if __name__ == '__main__':
    unittest.main()
