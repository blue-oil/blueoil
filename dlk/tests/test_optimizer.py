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
from core.data_types import Float32, PackedUint32, Int32, QUANTIZED_NOT_PACKED
from core.optimizer import pass_remove_identities, pass_transpose, pass_constant_folding, \
    pass_propagate_quantization_details_into_conv, pass_compute_thresholds, pass_pack_weights, \
    pass_quantize_convolutions, pass_propagate_datatypes, pass_propagate_output_type_backward
from core.graph import Graph
from core.operators import Add, AveragePool, BatchNormalization, Constant, Conv, Identity, Input, \
    MaxPool, Operator, Output, Transpose, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half, Reshape, Softmax, \
    SpaceToDepth

import numpy as np


class TestPassTranspose(unittest.TestCase):
    """Test class for transposing pass."""
    def test_pass_transpose(self) -> None:
        """Test code for transposing optimizer pass."""
        data = np.random.rand(3, 2, 2, 1)
        graph1 = self.create_sample_graph(data)
        graph2 = self.create_expected_graph(data)

        pass_transpose(graph1)

        self.assertEqual(graph1, graph2, 'transpose to NHWC failed.')

        print("Test pass #1 transpose passed!")

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

        # One output
        rs = Reshape('reshape', [1, 48], Float32(), {'data': conv})
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
        i1 = Identity('identity1', [1, 2, 2, 3], Float32(), {'input': w}, dimension_format='NHWC')
        q = QTZ_binary_mean_scaling('qtz1', [1, 2, 2, 3], Float32(), {'input': i1}, dimension_format='NHWC')

        # Conv
        conv = Conv('conv', [1, 4, 4, 3], Float32(), {'X': x, 'W': q}, kernel_shape=[2, 2], dimension_format='NHWC')

        # One output
        rs = Reshape('reshape', [1, 48], Float32(), {'data': conv})
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

        print("Test pass #2 remove identities passed!")

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

        # One output
        i2 = Identity('identity2', [1, 4, 4, 3], Float32(), {'input': conv})
        rs = Reshape('reshape', [1, 48], Float32(), {'data': i2})
        y = Output('output', [1, 48], Float32(), {'input': rs},)

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

        # One output
        rs = Reshape('reshape', [1, 48], Float32(), {'data': conv})
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

        print("Test pass #3 propagate_quantization_details_into_conv passed!")

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

        graph_2_1 = self.create_sample_graph_2(data1)
        graph_2_2 = self.create_sample_graph_2(data1)
        pass_pack_weights(graph_2_2)
        self.assertEqual(graph_2_1, graph_2_2,
                         '[Failed] Found optimized graph not the same')

        print("Test pass #4 pack_weights passed!")

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

    @staticmethod
    def create_sample_graph_2(data1: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], Float32(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        s1 = Constant('const1', Float32(), np.zeros([1, 4, 4, 3]))
        add1 = Add('add', [1, 4, 4, 3], Float32(), {'A': conv1, 'B': s1})

        y = Output('output', [1, 4, 4, 3], Float32(), {'input': add1})

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
        self.assertEqual(graph1.get_op('kqtz1').dtype, PackedUint32(),
                         '[Failed] Found output dtype of kernel quantizer not proper')
        self.assertEqual(graph1.get_op('conv2').dtype, Float32(),
                         '[Failed] Found output dtype of conv not proper')

        print("Test pass #5 quantize_convolutions passed!")

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

        print("Test pass #6 propagate data types passed!")

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

        pass_propagate_output_type_backward(graph1)

        self.assertEqual(graph1.get_op('conv1').dtype, Float32(),
                         '[Failed] Found dtype of SpaceToDepth not propagate correctly')

        print("Test pass #7 propagate output type backward passed!")

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


class TestPassComputeThresholds(unittest.TestCase):
    """Test class for packing weight."""
    def test_pass_compute_thresholds(self) -> None:
        """Test pass."""
        data1 = np.float32(np.random.rand(1, 2, 2, 3))
        data2 = np.float32(np.random.rand(1, 2, 2, 3))
        graph1 = self.create_sample_graph(data1, data2)

        pass_compute_thresholds(graph1)

        self.assertEqual(graph1.get_op('conv2').has_thresholds, True,
                         '[Failed] Found threshold of Conv not calculated')

        print("Test pass #8 compute_thresholds passed!")

    def test_pass_compute_thresholds_for_huge_threshold_values(self) -> None:
        """Test pass."""
        data1 = np.float32(np.random.rand(1, 2, 2, 3))
        data2 = np.float32(np.random.uniform(10 ** (-30), 10 ** (-40), size=(1, 2, 2, 3)))
        graph1 = self.create_sample_graph(data1, data2)

        pass_compute_thresholds(graph1)

        self.assertEqual(graph1.get_op('conv2').has_thresholds, True,
                         '[Failed] Found threshold of Conv not calculated')

        print("Test pass #8-1 compute_thresholds of enormous values passed!")

    @staticmethod
    def create_sample_graph(data1: np.ndarray, data2: np.ndarray) -> Graph:
        graph = Graph()

        # input
        x = Input('placeholder', [1, 5, 5, 3], Float32())

        # Conv1
        w1 = Constant('weight1', Float32(), data1)
        conv1 = Conv('conv1', [1, 4, 4, 3], Float32(), {'X': x, 'W': w1}, kernel_shape=[2, 2])

        # activation quantizer
        s1 = Constant('aq_const1', Int32(), np.array([2], dtype=np.int32))
        s2 = Constant('aq_const2', Float32(), np.array([2.0], dtype=np.float32))
        aq1 = QTZ_linear_mid_tread_half('aqtz1', [1, 4, 4, 3], Float32(), {'X': conv1, 'Y': s1, 'Z': s2})

        # Conv2
        w2 = Constant('weight2', Float32(), data2)
        kq = QTZ_binary_mean_scaling('kqtz1', [1, 2, 2, 3], Float32(), {'input': w2})
        conv2 = Conv('conv2', [1, 3, 3, 3], Float32(), {'X': aq1, 'W': kq}, kernel_shape=[2, 2])
        conv2.a_quantizer = [aq1]
        conv2.quantizer = kq
        conv2.is_quantized = True

        sc = Constant('bn_scale', Float32(), np.random.rand(3))
        be = Constant('bn_b', Float32(), np.random.rand(3))
        mu = Constant('bn_mu', Float32(), np.random.rand(3))
        va = Constant('bn_var', Float32(), np.random.rand(3))
        bn = BatchNormalization('bn', [1, 3, 3, 3], Float32(), {'X': conv2,
                                                                'scale': sc,
                                                                'B': be,
                                                                'mean': mu,
                                                                'var': va})

        # activation quantizer
        s3 = Constant('aq_const3', Int32(), np.array([2], dtype=np.int32))
        s4 = Constant('aq_const4', Float32(), np.array([2.0], dtype=np.float32))
        aq2 = QTZ_linear_mid_tread_half('aqtz2', [1, 3, 3, 3], Float32(), {'X': bn, 'Y': s3, 'Z': s4})

        # One output
        y = Output('output', [1, 3, 3, 3], Float32(), {'input': aq2})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


class TestPassConstantFolding(unittest.TestCase):
    """Test class for packing weight."""
    def test_pass_constant_folding(self) -> None:
        """Test pass."""
        graph1 = self.create_sample_graph()

        pass_constant_folding(graph1)

        self.assertEqual(set(graph1.get_op('potatoes_new').data), set(np.array([2, 5])),
                         '[Failed] Found folded constant not correct')

        print("Test pass #9 constant folding passed!")

    @staticmethod
    def create_sample_graph() -> Graph:
        graph = Graph()

        x = Input('placeholder', [2], Float32())

        s1 = Constant('potato_1', Float32(), np.array([1, 2]))
        s2 = Constant('potato_2', Float32(), np.array([1, 3]))
        add1 = Add('potatoes', [2], Float32(), {'A': s1, 'B': s2})
        add2 = Add('more_potatoes', [2], Float32(), {'A': x, 'B': add1})

        # One output
        y = Output('output', [2], Float32(), {'input': add2})

        # add ops to the graph
        graph.add_op_and_inputs(y)

        return graph


if __name__ == '__main__':
    unittest.main()
