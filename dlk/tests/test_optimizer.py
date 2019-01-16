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
    MaxPool, Operator, Output, Transpose, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half, Reshape, Softmax, \
    SpaceToDepth

import numpy as np
from typing import Tuple


class TestPassTranspose(unittest.TestCase):
    """Test class for transposing pass."""

    def test_pass_transpose(self) -> None:
        """Test code for transposing optimizer pass."""
        data1 = np.random.rand(3, 2, 2, 1)
        graph1 = self.create_sample_graph(data1)
        graph2 = self.create_expected_graph(data1)

        pass_transpose(graph1)

        self.assertEqual(graph1, graph2, 'transpose to NHWC failed.')

        print("Test transpose #1 pass passed!")

    @staticmethod
    def create_sample_graph(data: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [3, 5, 5, 1], Float32(), dimension_format='CWHN')

        # constant and internal nodes
        w = Constant('weight', Float32(), data, dimension_format='CWHN')
        i1 = Identity('identity1', [3, 2, 2, 1], Float32(), {'input': w}, dimension_format='CWHN')
        q = QTZ_binary_mean_scaling('qtz1', [3, 2, 2, 1], Float32(), {'input': i1}, dimension_format='CWHN')

        # Conv
        conv = Conv('conv', [3, 4, 4, 1], Float32(), {'X': x, 'W': q}, kernel_shape=[2, 2], dimension_format='CWHN')

        rs = Reshape('reshape', [1, 48], Float32(), {'data': conv})

        # One output
        y = Output('output', [1, 48], Float32(), {'input': rs},)

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    @staticmethod
    def create_expected_graph(data: np.ndarray) -> Graph:
        graph = Graph()

        data = data.transpose([3, 2, 1, 0])

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32(), dimension_format='NHWC')

        # constant and internal nodes
        w = Constant('weight', Float32(), data, dimension_format='NHWC')
        i = Identity('identity1', [1, 2, 2, 3], Float32(), {'input': w}, dimension_format='NHWC')
        q = QTZ_binary_mean_scaling('qtz1', [1, 2, 2, 3], Float32(), {'input': i}, dimension_format='NHWC')

        # Conv
        conv = Conv('conv', [1, 4, 4, 3], Float32(), {'X': x, 'W': q}, kernel_shape=[2, 2], dimension_format='NHWC')

        rs = Reshape('reshape', [1, 48], Float32(), {'data': conv})

        # One output
        y = Output('output', [1, 48], Float32(), {'input': rs},)

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassRemoveIdentities(unittest.TestCase):
    """Test class for removing identity pass."""

    def test_pass_remove_identities(self) -> None:
        """Test code for removing identities optimizer pass."""
        data = np.random.rand(1, 2, 2, 3)
        graph1 = self.create_sample_graph(data)
        graph2 = self.create_expected_graph(data)

        pass_remove_identities(graph1)

        self.assertEqual(graph1, graph2, 'remove identities failed.')

        print("Test remove identities #2 pass passed!")

    @staticmethod
    def create_sample_graph(data: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # constant and internal nodes
        w = Constant('weight', Float32(), data)
        i1 = Identity('identity1', [1, 2, 2, 3], Float32(), {'input': w})
        q = QTZ_binary_mean_scaling('qtz1', [1, 2, 2, 3], Float32(), {'input': i1})

        # Conv
        conv = Conv('conv', [1, 4, 4, 3], Float32(), {'X': x, 'W': q}, kernel_shape=[2, 2])

        i2 = Identity('identity2', [1, 4, 4, 3], Float32(), {'input': conv})

        rs = Reshape('reshape', [1, 48], Float32(), {'data': i2})

        # One output
        y = Output('output', [1, 48], Float32(), {'input': rs}, )

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    @staticmethod
    def create_expected_graph(data: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # constant and internal nodes
        w = Constant('weight', Float32(), data)
        q = QTZ_binary_mean_scaling('qtz1', [1, 2, 2, 3], Float32(), {'input': w})

        # Conv
        conv = Conv('conv', [1, 4, 4, 3], Float32(), {'X': x, 'W': q}, kernel_shape=[2, 2])

        rs = Reshape('reshape', [1, 48], Float32(), {'data': conv})

        # One output
        y = Output('output', [1, 48], Float32(), {'input': rs},)

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassPropagateQuantizationDetailsIntoConv(unittest.TestCase):
    """Test class for propagating quantization details into conv."""
    def test_pass_propagate_quantization_details_into_conv(self) -> None:
        """Test pass."""
        data1 = np.random.rand(1, 2, 2, 3)
        data2 = np.random.rand(1, 2, 2, 3)
        graph1 = self.create_sample_graph(data1, data2)
        graph2 = self.create_expected_graph(data1, data2)

        pass_propagate_quantization_details_into_conv(graph1)
        aq_g1 = graph1.get_op('conv2').a_quantizer
        aq_g2 = graph2.get_op('conv2').a_quantizer
        kq_g1 = graph1.get_op('conv2').quantizer
        kq_g2 = graph2.get_op('conv2').quantizer

        self.assertEqual(len(aq_g1), len(aq_g2), '[Failed] Found number of activation quantizer not matched')
        if aq_g1 and aq_g2:
            self.assertEqual(aq_g1[0].op_type, aq_g2[0].op_type,
                             '[Failed] Found type of activation quantizer not matched')
        self.assertEqual(kq_g1.op_type, kq_g2.op_type, '[Failed] Found type of kernel quantizer not matched')
        self.assertEqual(graph1, graph2, '[Failed] Expected graph not matched')

        print("Test propagate_quantization_details_into_conv #3 pass passed!")

    @staticmethod
    def create_sample_graph(data1: np.ndarray, data2: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], Float32(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        # activation quantizer
        s1 = Constant('aq_const1', Float32(), np.array(1))
        s2 = Constant('aq_const2', Float32(), np.array(2))
        aq = QTZ_linear_mid_tread_half('aqtz1', [1, 4, 4, 3], Float32(), {'X': conv1, 'Y': s1, 'Z': s2})

        # Conv2
        w2 = Constant('weight2', Float32(), data2)
        kq = QTZ_binary_mean_scaling('kqtz1', [1, 2, 2, 3], Float32(), {'input': w2})
        conv2 = Conv('conv2', [1, 3, 3, 3], Float32(), {'X': aq, 'W': kq}, kernel_shape=[2, 2])

        # One output
        y = Output('output', [1, 3, 3, 3], Float32(), {'input': conv2})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph

    @staticmethod
    def create_expected_graph(data1: np.ndarray, data2: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], Float32(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        # activation quantizer
        s1 = Constant('aq_const1', Float32(), np.array(1))
        s2 = Constant('aq_const2', Float32(), np.array(2))
        aq = QTZ_linear_mid_tread_half('aqtz1', [1, 4, 4, 3], Float32(), {'X': conv1, 'Y': s1, 'Z': s2})

        # Conv2
        w2 = Constant('weight2', Float32(), data2)
        kq = QTZ_binary_mean_scaling('kqtz1', [1, 2, 2, 3], Float32(), {'input': w2})
        conv2 = Conv('conv2', [1, 3, 3, 3], Float32(), {'X': aq, 'W': kq}, kernel_shape=[2, 2])
        conv2.a_quantizer = [aq]
        conv2.quantizer = kq

        # One output
        y = Output('output', [1, 3, 3, 3], Float32(), {'input': conv2})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassPackWeights(unittest.TestCase):
    """Test class for packing weight."""
    def test_pass_pack_weights(self) -> None:
        """Test pass."""
        data1 = np.float32(np.random.rand(1, 2, 2, 3))
        data2 = np.float32(np.random.rand(1, 2, 2, 3))
        graph1 = self.create_sample_graph(data1, data2)

        pass_pack_weights(graph1)

        self.assertEqual(graph1.get_op('conv2').input_ops['W'].op_type, 'Constant',
                         '[Failed] Found input kernel weights not a constant')

        print("Test pack_weights #4 pass passed!")

    @staticmethod
    def create_sample_graph(data1: np.ndarray, data2: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], Float32(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        # activation quantizer
        s1 = Constant('aq_const1', Float32(), np.array(1))
        s2 = Constant('aq_const2', Float32(), np.array(2))
        aq = QTZ_linear_mid_tread_half('aqtz1', [1, 4, 4, 3], Float32(), {'X': conv1, 'Y': s1, 'Z': s2})

        # Conv2
        w2 = Constant('weight2', Float32(), data2)
        kq = QTZ_binary_mean_scaling('kqtz1', [1, 2, 2, 3], Float32(), {'input': w2})
        conv2 = Conv('conv2', [1, 3, 3, 3], Float32(), {'X': aq, 'W': kq}, kernel_shape=[2, 2])
        conv2.a_quantizer = [aq]
        conv2.quantizer = kq

        # One output
        y = Output('output', [1, 3, 3, 3], Float32(), {'input': conv2})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassQuantizeConvolutions(unittest.TestCase):
    """Test class for packing weight."""
    def test_pass_quantize_convolutions(self) -> None:
        """Test pass."""
        data1 = np.float32(np.random.rand(1, 2, 2, 3))
        data2 = np.float32(np.random.rand(1, 2, 2, 3))
        graph1 = self.create_sample_graph(data1, data2)

        pass_quantize_convolutions(graph1)

        self.assertEqual(graph1.get_op('aqtz1').dtype, QUANTIZED_NOT_PACKED(),
                         '[Failed] Found output dtype of activation quantizer not proper')
        self.assertEqual(graph1.get_op('conv2').dtype, Float32(),
                         '[Failed] Found output dtype of conv not proper')

        print("Test quantize_convolutions #5 pass passed!")

    @staticmethod
    def create_sample_graph(data1: np.ndarray, data2: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], Float32(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        # activation quantizer
        s1 = Constant('aq_const1', Float32(), np.array(1))
        s2 = Constant('aq_const2', Float32(), np.array(2))
        aq = QTZ_linear_mid_tread_half('aqtz1', [1, 4, 4, 3], Float32(), {'X': conv1, 'Y': s1, 'Z': s2})

        # Conv2
        w2 = Constant('weight2', Float32(), data2)
        kq = QTZ_binary_mean_scaling('kqtz1', [1, 2, 2, 3], Float32(), {'input': w2})
        conv2 = Conv('conv2', [1, 3, 3, 3], Float32(), {'X': aq, 'W': kq}, kernel_shape=[2, 2])
        conv2.a_quantizer = [aq]
        conv2.quantizer = kq

        # One output
        y = Output('output', [1, 3, 3, 3], Float32(), {'input': conv2})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassPropagateDatatypes(unittest.TestCase):
    """Test class for packing weight."""
    def test_pass_propagate_datatypes(self) -> None:
        """Test pass."""
        data1 = np.float32(np.random.rand(1, 2, 2, 3))
        graph1 = self.create_sample_graph(data1)
        # graph2 = self.create_expected_graph(data1, data2)

        pass_propagate_datatypes(graph1)

        self.assertEqual(graph1.get_op('s2d').dtype, QUANTIZED_NOT_PACKED(),
                         '[Failed] Found dtype of SpaceToDepth not propagate correctly')

        print("Test propagate datatypes #6 pass passed!")

    @staticmethod
    def create_sample_graph(data1: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], QUANTIZED_NOT_PACKED(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        pool1 = SpaceToDepth('s2d', [1, 2, 2, 12], Float32(), {'input': conv1})

        # One output
        y = Output('output', [1, 2, 2, 12], Float32(), {'input': pool1})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassPropagateOutputTypeBackward(unittest.TestCase):
    """Test class for packing weight."""
    def test_pass_propagate_output_type_backward(self) -> None:
        """Test pass."""
        data1 = np.float32(np.random.rand(1, 2, 2, 3))
        graph1 = self.create_sample_graph(data1)
        # graph2 = self.create_expected_graph(data1, data2)

        pass_propagate_output_type_backward(graph1)

        self.assertEqual(graph1.get_op('conv1').dtype, Float32(),
                         '[Failed] Found dtype of SpaceToDepth not propagate correctly')

        print("Test propagate output type backward #7 pass passed!")

    @staticmethod
    def create_sample_graph(data1: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], QUANTIZED_NOT_PACKED(), {'X': x, 'W': w1}, kernel_shape=[2, 2])
        conv1.is_quantized = True

        pool1 = SpaceToDepth('s2d', [1, 2, 2, 12], Float32(), {'input': conv1})

        # One output
        y = Output('output', [1, 2, 2, 12], Float32(), {'input': pool1})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


if __name__ == '__main__':
    unittest.main()
