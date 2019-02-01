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
"""Test file for OnnxIO."""
import functools
import unittest
from os import path, makedirs
from typing import List

import numpy as np

from core import Operator
from core.data_types import Float32
from core.graph import Graph
from core.model import Model
from core.operators import Input, Output, Constant, Conv
from frontend import TensorFlowIO
from scripts.simple_model import make_simple_model as make_model


class TestTensorFlowIO(unittest.TestCase):
    """Test class for TensorflowIO."""

    def test_tf_import(self) -> None:
        """Test code for importing Tensorflow file with TensorflowIO."""
        tf_path = path.join('examples',
                            'classification',
                            'lmnet_v1_toy_graph',
                            'minimal_graph_with_shape.pb')

        tf_io = TensorFlowIO()
        model = tf_io.read(tf_path)

        graph: Graph = model.graph
        outputs = graph.get_outputs()

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, [1, 10])

        print("TF file import test passed!")

    def test_import_classification(self) -> None:
        """Test code for importing Tensorflow file with TensorflowIO."""
        tf_path = path.join('examples',
                            'classification',
                            'lmnet_quantize_cifar10_stride_2.20180523.3x3',
                            'minimal_graph_with_shape.pb')

        tf_io = TensorFlowIO()
        model = tf_io.read(tf_path)

        graph: Graph = model.graph
        outputs = graph.get_outputs()

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, [1, 10])

        print("TF file import test passed for classification!")

    def test_import_group_convolution_classification(self) -> None:
        """Test code for importing Tensorflow file with TensorflowIO."""
        tf_path = path.join('examples',
                            'classification',
                            'lmnet_v1_group_conv',
                            'minimal_graph_with_shape.pb')

        tf_io = TensorFlowIO()
        model = tf_io.read(tf_path)

        graph: Graph = model.graph
        outputs = graph.get_outputs()

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, [1, 10])

        print("TF file import test passed for group convolution!")

    def test_import_object_detection_lack_shape_info(self) -> None:
        """Test code for importing tf pb file of object detection
        (lack of shape info for some operator) with TensorflowIO.
        """
        tf_path = path.join('examples',
                            'object_detection',
                            'fyolo_quantize_17_v14_some_1x1_wide_pact_add_conv',
                            'minimal_graph_with_shape.pb')

        tf_io = TensorFlowIO()
        model = tf_io.read(tf_path)

        graph: Graph = model.graph
        outputs = graph.get_outputs()

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, [1, 10, 10, 125])

        print("TF file import test passed for object detection!")

    def make_simple_model(self) -> Model:
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
            np.zeros([1, 2, 2, 3]),
            dimension_format='NHWC',
        )

        # Conv
        conv = Conv(
            'conv',
            [1, 4, 4, 1],
            Float32(),
            {'X': x, 'W': w},
            kernel_shape=[2, 2]
        )

        # One output
        y = Output(
            'output',
            [1, 4, 4, 1],
            Float32(),
            {'input': conv}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)
        model = Model()
        model.graph = graph
        return model

    def _comparator(self, graph_1, graph_2) -> bool:
        if graph_1 is None or graph_2 is None or not isinstance(graph_2, Graph) or not isinstance(graph_1, Graph):
            return False

        def equals(op1: Operator, op2: Operator) -> bool:
            """Return if these two objects are equivalent."""
            if op1 is None or not isinstance(op1, Operator):
                print(f'{op1.name} has different type.')
                return False

            if op2 is None or not isinstance(op2, Operator):
                print(f'{op2.name} has different type.')
                return False

            eq_type = op1.op_type == op2.op_type
            if not eq_type:
                print(f'{op1.name} and {op2.name} have different type: {op1.op_type} and {op2.op_type}')

            eq_dtype = op1.dtype == op2.dtype
            if not eq_dtype:
                print(f'{op1.name} and {op2.name} have different dtype: {op1.dtype} and {op2.dtype}')

            # Transpose the graph for data comparison if necessary
            if op1.dimension != op2.dimension:
                perm = [op2.dimension.index(s) for s in op1.dimension]
                new_shape: List[int] = [op2.shape[i] for i in perm]
                new_dimension: str = functools.reduce(lambda x, y: x + y, [op2.dimension[i] for i in perm])
                new_data: np.ndarray = op2.data.transpose(perm)
            else:
                new_shape = op2.shape
                new_dimension = op2.dimension
                new_data = op2.data

            eq_dim = op1.dimension == new_dimension
            if not eq_dim:
                print(f'{op1.name} and {op2.name} have different dimension: {op1.dimension} and {op2.dimension}')

            eq_shape = op1.shape == new_shape
            if not eq_shape:
                print(f'{op1.name} and {op2.name} have different shape: {op1.shape} and {op2.shape}')

            eq_data = eq_shape and np.allclose(op1.data, new_data)
            if not eq_data:
                print(f'{op1.name} and {op2.name} have different data: {op1.data} and {op2.data}')

            return eq_type and eq_shape and eq_dtype and eq_dim and eq_data

        def match(op1: Operator, op2: Operator) -> bool:
            if not equals(op1, op2):
                print(f'{op1.name} is different.')
                return False

            # check input nodes and further
            for i1, i2 in zip(op1.input_ops.values(), op2.input_ops.values()):
                if not match(i1, i2):
                    return False
            return True

        for o1, o2 in zip(graph_1.get_outputs(), graph_2.get_outputs()):
            if not match(o1, o2):
                return False
        return True


if __name__ == '__main__':
    unittest.main()
