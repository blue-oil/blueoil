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
"""Test file for Optimizer."""
import unittest
from core.data_types import Float32, Uint32, Int32, QUANTIZED_NOT_PACKED
from core.optimizer import pass_remove_identities, pass_transpose, pass_constant_folding, \
    pass_propagate_quantization_details_into_conv, pass_compute_thresholds, pass_pack_weights, \
    pass_quantize_convolutions, pass_propagate_datatypes, pass_propagate_output_type_backward
from core.graph import Graph
from core.operators import Add, AveragePool, BatchNormalization, Constant, Conv, Identity, Input, \
    MaxPool, Operator, Output, Transpose, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half, Reshape, Softmax

import numpy as np
from typing import Tuple


class TestOptimizer(unittest.TestCase):
    """Test class for GraphRunner."""

    def test_precompute1(self) -> None:
        """Test code for precompute optimizer."""
        data1 = np.random.rand(3, 2, 2, 3)
        data2 = np.random.rand(3, 2, 2, 3)
        data3 = np.random.rand(3, 2, 2, 3)
        graph1 = self.create_sample_graph(data1, data2, data3)
        graph2 = self.create_precompute_graph(data1, data2, data3)

        pass_remove_identities(graph1)
        pass_transpose(graph1)

        pass_constant_folding(graph1)

        self.assertEqual(graph1, graph2, 'precompute failed.')

        print("Precompute test #1 passed!")

    def test_precompute2(self) -> None:
        """Test code for precompute optimizer."""
        data1 = np.random.rand(3, 2, 2, 3)
        data2 = np.random.rand(3, 2, 2, 3)
        data3 = np.random.rand(3, 2, 2, 3)
        graph1 = self.create_sample_graph(data1, data2, data3)
        graph2, scaling1, scaling2 = self.create_quantized_graph(data1, data2, data3)

        pass_remove_identities(graph1)
        pass_transpose(graph1)

        pass_propagate_quantization_details_into_conv(graph1)
        pass_pack_weights(graph1)
        pass_quantize_convolutions(graph1)

        pass_propagate_datatypes(graph1)

        pass_constant_folding(graph1)

        self.assertEqual(graph1, graph2, 'precompute failed.')
        self.assertAlmostEqual(graph1.get_op('conv2').quantizer.scaling_factor, scaling2)  # type: ignore

        print("Precompute test #2 passed!")

    def test_precompute3(self) -> None:
        """Test code for precompute optimizer."""
        data1 = np.random.rand(3, 2, 2, 3)
        data2 = np.random.rand(3, 2, 2, 3)
        data3 = np.random.rand(3, 2, 2, 3)
        graph1 = self.create_sample_graph3(data1, data2, data3)
        graph2, scaling2, scaling3 = self.create_quantized_graph2(data1, data2, data3)

        pass_remove_identities(graph1)
        pass_transpose(graph1)

        pass_propagate_quantization_details_into_conv(graph1)
        pass_pack_weights(graph1)
        pass_quantize_convolutions(graph1)

        pass_propagate_datatypes(graph1)

        pass_constant_folding(graph1)

        self.assertEqual(graph1, graph2, 'precompute failed.')
        self.assertAlmostEqual(graph1.get_op('conv2').quantizer.scaling_factor, scaling2)  # type: ignore
        self.assertAlmostEqual(graph1.get_op('conv3').quantizer.scaling_factor, scaling3)  # type: ignore

        print("Precompute test #3 passed!")

    def test_transpose_NHWC(self) -> None:
        """Test code for transpose_NHWC optimizer."""
        data = np.random.rand(3, 2, 2, 1)
        graph1 = self.create_sample_graph2(data)
        graph2 = self.create_transposed_graph(data)

        pass_transpose(graph1)

        self.assertEqual(graph1, graph2, 'transpose to NHWC failed.')

        print("Transpose_NHWC test #1 passed!")

    def create_sample_graph(self, data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input(
            'placeholder',
            [1, 5, 5, 3],
            Float32(),
        )

        # constant and internal nodes
        w = Constant(
            'weight',
            Float32(),
            data1
        )

        i = Identity(
            'identity1',
            [3, 2, 2, 3],
            Float32(),
            {'input': w}
        )

        t = Transpose(
            'transpose1',
            [3, 2, 2, 3],
            Float32(),
            {'data': i},
            perm=[3, 2, 1, 0]
        )

        q = QTZ_binary_mean_scaling(
            'qtz1',
            [3, 2, 2, 3],
            Float32(),
            {'input': t}
        )

        # Conv
        conv1 = Conv(
            'conv1',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': q},
            kernel_shape=[2, 2]
        )

        i2 = Identity(
            'identity2',
            [1, 4, 4, 3],
            Float32(),
            {'input': conv1}
        )

        s1 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s2 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq = QTZ_linear_mid_tread_half(
            'aqtz1',
            [1, 4, 4, 3],
            Float32(),
            {'X': i2, 'Y': s1, 'Z': s2}
        )

        dummy = Transpose(
            'dummy',
            [1, 4, 4, 3],
            Float32(),
            {'data': aq},
            perm=[0, 1, 2, 3]
        )

        w2 = Constant(
            'weight2',
            Float32(),
            data2
        )

        q2 = QTZ_binary_mean_scaling(
            'qtz2',
            [3, 2, 2, 3],
            Float32(),
            {'input': w2}
        )

        conv2 = Conv(
            'conv2',
            [1, 3, 3, 3],
            Float32(),
            {'X': dummy, 'W': q2},
            kernel_shape=[2, 2]
        )

        s3 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s4 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq2 = QTZ_linear_mid_tread_half(
            'aqtz2',
            [1, 3, 3, 3],
            Float32(),
            {'X': conv2, 'Y': s3, 'Z': s4}
        )

        w3 = Constant(
            'weight3',
            Float32(),
            data3
        )

        i3 = Identity(
            'identity3',
            [1, 3, 3, 3],
            Float32(),
            {'input': aq2}
        )

        conv3 = Conv(
            'conv3',
            [1, 2, 2, 3],
            Float32(),
            {'X': i3, 'W': w3},
            kernel_shape=[2, 2]
        )

        # One output
        y = Output(
            'output',
            [1, 2, 2, 3],
            Float32(),
            {'input': conv3}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    def binary_mean_scaling(self, data: np.ndarray) -> Tuple[np.float32, np.ndarray]:
        return np.mean(np.abs(data)), np.sign(data).astype(np.float32)

    def create_precompute_graph(self, data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> Graph:
        graph = Graph()

        # two inputs
        x = Input(
            'placeholder',
            [1, 5, 5, 3],
            Float32(),
        )

        scaling1, qdata = self.binary_mean_scaling(data1.transpose([3, 2, 1, 0]))
        w = Constant(
            'weight',
            Float32(),
            qdata * scaling1
        )

        # Conv
        conv1 = Conv(
            'conv1',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': w},
            kernel_shape=[2, 2]
        )

        s1 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s2 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq = QTZ_linear_mid_tread_half(
            'aqtz1',
            [1, 4, 4, 3],
            Float32(),
            {'X': conv1, 'Y': s1, 'Z': s2}
        )

        dummy = Transpose(
            'dummy',
            [1, 4, 4, 3],
            Float32(),
            {'data': aq},
            perm=[0, 1, 2, 3]
        )

        scaling2, qdata2 = self.binary_mean_scaling(data2)
        w2 = Constant(
            'weight2',
            Float32(),
            qdata2 * scaling2
        )

        conv2 = Conv(
            'conv2',
            [1, 3, 3, 3],
            Float32(),
            {'X': dummy, 'W': w2},
            kernel_shape=[2, 2]
        )

        s3 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s4 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq2 = QTZ_linear_mid_tread_half(
            'aqtz2',
            [1, 3, 3, 3],
            Float32(),
            {'X': conv2, 'Y': s3, 'Z': s4}
        )

        w3 = Constant(
            'weight3',
            Float32(),
            data3
        )

        conv3 = Conv(
            'conv3',
            [1, 2, 2, 3],
            Float32(),
            {'X': aq2, 'W': w3},
            kernel_shape=[2, 2]
        )

        # One output
        y = Output(
            'output',
            [1, 2, 2, 3],
            Float32(),
            {'input': conv3}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    def create_quantized_graph(self, data: np.ndarray, data2: np.ndarray, data3: np.ndarray) \
            -> Tuple[Graph, np.float32, np.float32]:
        graph = Graph()

        # two inputs
        x = Input(
            'placeholder',
            [1, 5, 5, 3],
            Float32(),
        )

        from modules.packer import Packer
        packer = Packer(1, 32)
        data = data.transpose([3, 2, 1, 0])
        scaling, qdata = self.binary_mean_scaling(data)
        shape = list(data.shape)
        w = Constant(
            'weight',
            Float32(),
            qdata * scaling,
        )

        q = QTZ_binary_mean_scaling(
            'qtz1',
            shape,
            Float32(),
            {'input': w}
        )
        q.scaling_factor = scaling

        # Conv
        conv1 = Conv(
            'conv1',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': w},
            kernel_shape=[2, 2],
        )

        s1 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s2 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq = QTZ_linear_mid_tread_half(
            'aqtz1',
            [1, 4, 4, 3],
            QUANTIZED_NOT_PACKED(),
            {'X': conv1, 'Y': s1, 'Z': s2}
        )

        dummy = Transpose(
            'dummy',
            [1, 4, 4, 3],
            QUANTIZED_NOT_PACKED(),
            {'data': aq},
            perm=[0, 1, 2, 3]
        )

        scaling2, qdata2 = self.binary_mean_scaling(data2)
        w2 = Constant(
            'weight2',
            Uint32(),
            packer.run(qdata2),
            packed=True,
            actual_shape=[3, 2, 2, 3]
        )

        # quantizer connected to conv2 as 'conv2.quantizer'
        q2 = QTZ_binary_mean_scaling(
            'qtz2',
            [3, 2, 2, 3],
            Uint32(),
            {'input': w2}
        )
        q2.scaling_factor = scaling2

        conv2 = Conv(
            'conv2',
            [1, 3, 3, 3],
            Float32(),
            {'X': dummy, 'W': w2},
            kernel_shape=[2, 2],
            quantized=True
        )
        conv2.quantizer = q2

        s3 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s4 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq2 = QTZ_linear_mid_tread_half(
            'aqtz2',
            [1, 3, 3, 3],
            Float32(),
            {'X': conv2, 'Y': s3, 'Z': s4}
        )

        w3 = Constant(
            'weight3',
            Float32(),
            data3
        )

        conv3 = Conv(
            'conv3',
            [1, 2, 2, 3],
            Float32(),
            {'X': aq2, 'W': w3},
            kernel_shape=[2, 2]
        )

        # One output
        y = Output(
            'output',
            [1, 2, 2, 3],
            Float32(),
            {'input': conv3}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph, scaling, scaling2

    def create_sample_graph2(self, data: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input(
            'placeholder',
            [3, 5, 5, 1],
            Float32(),
            dimension_format='CWHN'
        )

        # constant and internal nodes
        w = Constant(
            'weight',
            Float32(),
            data,
            dimension_format='CWHN'
        )

        i = Identity(
            'identity1',
            [3, 2, 2, 1],
            Float32(),
            {'input': w},
            dimension_format='CWHN'
        )

        q = QTZ_binary_mean_scaling(
            'qtz1',
            [3, 2, 2, 1],
            Float32(),
            {'input': i},
            dimension_format='CWHN'
        )

        # Conv
        conv = Conv(
            'conv',
            [3, 4, 4, 1],
            Float32(),
            {'X': x, 'W': q},
            kernel_shape=[2, 2],
            dimension_format='CWHN'
        )

        rs = Reshape(
            'reshape',
            [1, 48],
            Float32(),
            {'data': conv}
        )

        # One output
        y = Output(
            'output',
            [1, 48],
            Float32(),
            {'input': rs},
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    def create_transposed_graph(self, data: np.ndarray) -> Graph:
        graph = Graph()
        data = data.transpose([3, 2, 1, 0])

        # input
        x = Input(
            'placeholder',
            [1, 5, 5, 3],
            Float32(),
            dimension_format='NHWC'
        )

        # constant and internal nodes
        w = Constant(
            'weight',
            Float32(),
            data,
            dimension_format='NHWC'
        )

        i = Identity(
            'identity1',
            [1, 2, 2, 3],
            Float32(),
            {'input': w},
            dimension_format='NHWC'
        )

        q = QTZ_binary_mean_scaling(
            'qtz1',
            [1, 2, 2, 3],
            Float32(),
            {'input': i},
            dimension_format='NHWC'
        )

        # Conv
        conv = Conv(
            'conv',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': q},
            kernel_shape=[2, 2],
            dimension_format='NHWC'
        )

        rs = Reshape(
            'reshape',
            [1, 48],
            Float32(),
            {'data': conv}
        )

        # One output
        y = Output(
            'output',
            [1, 48],
            Float32(),
            {'input': rs},
        )

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    def create_sample_graph3(self, data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input(
            'placeholder',
            [1, 5, 5, 3],
            Float32(),
        )

        # constant and internal nodes
        w = Constant(
            'weight',
            Float32(),
            data1
        )

        q = QTZ_binary_mean_scaling(
            'qtz1',
            [3, 2, 2, 3],
            Float32(),
            {'input': w}
        )

        # Conv
        conv1 = Conv(
            'conv1',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': q},
            kernel_shape=[2, 2]
        )

        i2 = Identity(
            'identity2',
            [1, 4, 4, 3],
            Float32(),
            {'input': conv1}
        )

        s1 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s2 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq = QTZ_linear_mid_tread_half(
            'aqtz1',
            [1, 4, 4, 3],
            Float32(),
            {'X': i2, 'Y': s1, 'Z': s2}
        )

        w2 = Constant(
            'weight2',
            Float32(),
            data2
        )

        q2 = QTZ_binary_mean_scaling(
            'qtz2',
            [3, 2, 2, 3],
            Float32(),
            {'input': w2}
        )

        conv2 = Conv(
            'conv2',
            [1, 3, 3, 3],
            Float32(),
            {'X': aq, 'W': q2},
            kernel_shape=[2, 2]
        )

        w3 = Constant(
            'weight3',
            Float32(),
            data3
        )

        q3 = QTZ_binary_mean_scaling(
            'qtz3',
            [3, 2, 2, 3],
            Float32(),
            {'input': w3}
        )

        conv3 = Conv(
            'conv3',
            [1, 3, 3, 3],
            Float32(),
            {'X': aq, 'W': q3},
            kernel_shape=[2, 2]
        )

        y1 = Output(
            'output1',
            [1, 3, 3, 3],
            Float32(),
            {'input': conv2}
        )

        y2 = Output(
            'output2',
            [1, 3, 3, 3],
            Float32(),
            {'input': conv3}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y1)
        graph.add_op_and_inputs(y2)

        return graph

    def create_quantized_graph2(self, data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input(
            'placeholder',
            [1, 5, 5, 3],
            Float32(),
        )

        # constant and internal nodes
        scaling1, qdata1 = self.binary_mean_scaling(data1)
        w = Constant(
            'weight',
            Float32(),
            qdata1 * scaling1
        )

        q = QTZ_binary_mean_scaling(
            'qtz1',
            [3, 2, 2, 3],
            Float32(),
            {'input': w}
        )

        # Conv
        conv1 = Conv(
            'conv1',
            [1, 4, 4, 3],
            Float32(),
            {'X': x, 'W': w},
            kernel_shape=[2, 2]
        )

        s1 = Constant(
            'aq_const1',
            Float32(),
            np.array(1)
        )

        s2 = Constant(
            'aq_const2',
            Float32(),
            np.array(2)
        )

        aq = QTZ_linear_mid_tread_half(
            'aqtz1',
            [1, 4, 4, 3],
            QUANTIZED_NOT_PACKED(),
            {'X': conv1, 'Y': s1, 'Z': s2}
        )

        from modules.packer import Packer
        packer = Packer(1, 32)
        scaling2, qdata2 = self.binary_mean_scaling(data2)
        w2 = Constant(
            'weight2',
            Uint32(),
            packer.run(qdata2),
            packed=True,
            actual_shape=[3, 2, 2, 3]
        )

        q2 = QTZ_binary_mean_scaling(
            'qtz2',
            [3, 2, 2, 3],
            Float32(),
            {'input': w2}
        )
        q2.scaling_factor = scaling2

        conv2 = Conv(
            'conv2',
            [1, 3, 3, 3],
            Float32(),
            {'X': aq, 'W': w2},
            kernel_shape=[2, 2],
            quantized=True,
        )
        conv2.quantizer = q2

        scaling3, qdata3 = self.binary_mean_scaling(data3)
        w3 = Constant(
            'weight2',
            Uint32(),
            packer.run(qdata3),
            packed=True,
            actual_shape=[3, 2, 2, 3]
        )

        q3 = QTZ_binary_mean_scaling(
            'qtz3',
            [3, 2, 2, 3],
            Float32(),
            {'input': w3}
        )
        q3.scaling_factor = scaling3

        conv3 = Conv(
            'conv3',
            [1, 3, 3, 3],
            Float32(),
            {'X': aq, 'W': w3},
            kernel_shape=[2, 2],
            quantized=True
        )
        conv3.quantizer = q3

        y1 = Output(
            'output1',
            [1, 3, 3, 3],
            Float32(),
            {'input': conv2}
        )

        y2 = Output(
            'output2',
            [1, 3, 3, 3],
            Float32(),
            {'input': conv3}
        )

        # add ops to the graph
        graph.add_op_and_inputs(y1)
        graph.add_op_and_inputs(y2)

        return graph, scaling2, scaling3


if __name__ == '__main__':
    unittest.main()
