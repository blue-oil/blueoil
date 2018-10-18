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
"""Test file for GraphRunner."""
import unittest
from core.data_types import Float32
from core.graph import Graph, GraphRunner
from core.operators import Conv, Input, Output, Constant, Operator
import numpy as np
from typing import Any, Dict, List


class TestRunner(GraphRunner):
    """Test class of GraphRunner.

    This just list up all the op_type of the graph.
    """

    def __init__(self, graph: Graph, depth_first: bool = True, lazy: bool = True) -> None:
        self.message: List[str] = []
        super().__init__(graph, depth_first=depth_first, lazy=lazy)

    def initialize(self, **kwargs: Any) -> None:
        self.message.append('start running.')

    def finalize(self, **kwargs: Any) -> None:
        self.message.append('finished running.')

    # backward: ouput -> inputs
    def run_backward_by_default(self, node: Operator, **kwargs: Any) -> None:
        kwargs['backward'].append(node.name)

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        self.message.append(f'{node.name}: backward process')
        super().run_backward_conv(node, **kwargs)

    # forward: inputs -> output
    def run_forward_by_default(self, node: Operator, **kwargs: Any) -> None:
        kwargs['forward'].append(node.name)

    def run_forward_conv(self, node: Conv, **kwargs: Any) -> None:
        self.message.append(f'{node.name}: forward process')
        super().run_forward_conv(node, **kwargs)


class TestGraphRunner(unittest.TestCase):
    """Test class for GraphRunner."""

    def test_graphrunner_default(self) -> None:
        """Test code for GraphRunner, with the depth-first mode (default)."""
        graph = Graph()
        self.create_graph(graph)

        kwargs: Dict[str, List[str]] = {'backward': [], 'forward': []}
        runner = TestRunner(graph)
        runner.run(**kwargs)

        lst1 = ['output', 'conv4', 'input3', 'conv3', 'input2', 'conv2', 'conv1', 'input1', 'weight1', 'weight2']
        self.assertEqual(kwargs['backward'], lst1,
                         'backward traversal failed in depth-first mode.')

        lst2 = ['input3', 'input2', 'input1', 'weight1', 'conv1', 'weight2', 'conv2', 'conv3', 'conv4', 'output']
        self.assertEqual(kwargs['forward'], lst2, 'forward traversal failed in depth-first mode.')

        self.assertEqual(runner.message, [
            'start running.',
            'conv4: backward process',
            'conv3: backward process',
            'conv2: backward process',
            'conv1: backward process',
            'conv1: forward process',
            'conv2: forward process',
            'conv3: forward process',
            'conv4: forward process',
            'finished running.',
        ])

        print("GraphRunner depth-first mode test passed!")

    def test_graphrunner_breadth_first(self) -> None:
        """Test code for GraphRunner, with the breadth-first mode."""
        graph = Graph()
        self.create_graph(graph)

        kwargs: Dict[str, List[str]] = {'backward': [], 'forward': []}
        runner = TestRunner(graph, depth_first=False, lazy=False)
        runner.run(**kwargs)

        lst1 = ['output', 'conv4', 'input3', 'conv3', 'input2', 'conv2', 'conv1', 'weight2', 'input1', 'weight1']
        self.assertEqual(kwargs['backward'], lst1,
                         'backward traversal failed in breadth-first mode.')

        lst2 = ['input3', 'input2', 'input1', 'weight1', 'weight2',
                'conv4', 'conv3', 'conv1', 'conv2', 'output']
        self.assertEqual(kwargs['forward'], lst2, 'forward traversal failed in breadth-first mode.')

        self.assertEqual(runner.message, [
            'start running.',
            'conv4: backward process',
            'conv3: backward process',
            'conv2: backward process',
            'conv1: backward process',
            'conv4: forward process',
            'conv3: forward process',
            'conv1: forward process',
            'conv2: forward process',
            'finished running.',
        ])

        print("GraphRunner bradth-first mode test passed!")

    def test_graphrunner_lazy_breadth_first(self) -> None:
        """Test code for GraphRunner, with the lazy breadth-first mode."""
        graph = Graph()
        self.create_graph(graph)

        kwargs: Dict[str, List[str]] = {'backward': [], 'forward': []}
        runner = TestRunner(graph, depth_first=False, lazy=True)
        runner.run(**kwargs)

        lst1 = ['output', 'conv4', 'input3', 'conv3', 'input2', 'conv2', 'conv1', 'weight2', 'input1', 'weight1']
        self.assertEqual(kwargs['backward'], lst1,
                         'backward traversal failed in breadth-first mode.')

        lst2 = ['input3', 'input2', 'input1', 'weight1', 'weight2',
                'conv1', 'conv2', 'conv3', 'conv4', 'output']
        self.assertEqual(kwargs['forward'], lst2, 'forward traversal failed in breadth-first mode.')

        self.assertEqual(runner.message, [
            'start running.',
            'conv4: backward process',
            'conv3: backward process',
            'conv2: backward process',
            'conv1: backward process',
            'conv1: forward process',
            'conv2: forward process',
            'conv3: forward process',
            'conv4: forward process',
            'finished running.',
        ])

        print("GraphRunner lazy breadth-first mode test passed!")

    def create_graph(self, graph):

        x1 = Input(
            'input1',
            [1, 4, 4, 3],
            Float32(),
        )

        w1 = Constant(
            'weight1',
            Float32(),
            np.zeros([1, 2, 2, 3])
        )

        conv1 = Conv(
            'conv1',
            [1, 3, 3, 3],
            Float32(),
            {'X': x1, 'W': w1},
            kernel_shape=[2, 2]
        )

        w2 = Constant(
            'weight2',
            Float32(),
            np.zeros([3, 2, 2, 3])
        )

        conv2 = Conv(
            'conv2',
            [1, 2, 2, 3],
            Float32(),
            {'X': conv1, 'W': w2},
            kernel_shape=[2, 2]
        )

        x2 = Input(
            'input2',
            [3, 3, 3, 3],
            Float32(),
        )

        x3 = Input(
            'input3',
            [3, 3, 3, 3],
            Float32(),
        )

        conv3 = Conv(
            'conv3',
            [3, 2, 2, 3],
            Float32(),
            {'X': x2, 'W': conv2},
            kernel_shape=[2, 2]
        )

        conv4 = Conv(
            'conv4',
            [1, 2, 2, 3],
            Float32(),
            {'X': x3, 'W': conv3},
            kernel_shape=[2, 2]
        )

        y = Output(
            'output',
            [1, 2, 2, 3],
            Float32(),
            {'input': conv4}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)


if __name__ == '__main__':
    unittest.main()
