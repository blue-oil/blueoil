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
"""Module of optimization passes."""
import numpy as np
from core.data_types import DataType, Float32, Float64, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64, \
    Bool, String, QUANTIZED_NOT_PACKED
from core.graph import Graph, GraphRunner
from core.operators import Add, AveragePool, BatchNormalization, Constant, Conv, Identity, Input, \
    MaxPool, Operator, Output, Transpose, Quantizer, QTZ_binary_mean_scaling, QTZ_linear_mid_tread_half, \
    Reshape, Softmax, Relu, Flatten, Dropout, Gemm, SpaceToDepth, QTZ_binary_channel_wise_mean_scaling, ConcatOnDepth,\
    Maximum, DepthToSpace, Split

from typing import Any, Dict, List, Optional, Set, cast
from functools import reduce
from enum import Enum

from modules.packer import Packer

NodeGroup = List[Operator]


def node_is_add(node: Operator) -> bool:
    return node.op_type == 'Add'


def node_is_conv(node: Operator) -> bool:
    return node.op_type == 'Conv'


def node_is_concat(node: Operator) -> bool:
    return node.op_type == 'ConcatV2'


def node_is_const(node: Operator) -> bool:
    return node.op_type == 'Constant'


def node_is_qconv(node: Operator) -> bool:
    return node.op_type == 'Conv' and cast(Conv, node).is_quantized


def node_is_input(node: Operator) -> bool:
    return node.op_type == 'Input'


def node_is_weight_quantizer(node: Operator) -> bool:
    return (node.op_type == 'QTZ_binary_mean_scaling'
            or node.op_type == 'QTZ_binary_channel_wise_mean_scaling')


def node_is_activation_quantizer(node: Operator) -> bool:
    return node.op_type == 'QTZ_linear_mid_tread_half'


def transpose_kernels(kernel_data, dimension_format, oh, ow, od, kh, kw, kd):
    NUM_PE          = 16
    NBIT_QDYPE      = 32
    MAX_NBIT_QINPUT = 2
    MAX_NBIT_KERNEL = 1
    num_qinput_per_qword    = int(NBIT_QDYPE / MAX_NBIT_QINPUT)
    num_qkernel_per_qword   = int(NBIT_QDYPE / MAX_NBIT_KERNEL)
    k_c_by_word             = int((kd + (num_qkernel_per_qword - 1)) / num_qkernel_per_qword);
    k_n_aligned_with_num_pe = int(((od + (NUM_PE - 1)) / NUM_PE) * NUM_PE);
    if od < NUM_PE:
        k_size = k_n_aligned_with_num_pe * kh * kw * k_c_by_word;
    else:
        k_size = od * kh * kw * k_c_by_word;

    flatten_value = []
    for elem in kernel_data:
        flatten_value.extend(elem)
    copy_value = [0] * k_size
    for i in range(od * kh * kw * k_c_by_word):
        copy_value[i] = flatten_value[i]

    transpose_values = [0] * k_size
    if (od < NUM_PE):
        kn_out = int(k_n_aligned_with_num_pe / NUM_PE)
    else:
        kn_out = int(od / NUM_PE)
    idx_src = 0

    if dimension_format == "NHWC":
        for no in range(kn_out):
            for ni in range(NUM_PE):
                for h in range(kh):
                    for w in range(kw):
                        for c in range(k_c_by_word):
                            idx_dst = h * (kw * kn_out * k_c_by_word * NUM_PE)
                            idx_dst += w * (kn_out * k_c_by_word * NUM_PE)
                            idx_dst += no * (k_c_by_word * NUM_PE)
                            idx_dst += c * (NUM_PE)
                            idx_dst += ni
                            transpose_values[idx_dst] = copy_value[idx_src]
                            idx_src += 1
    elif dimension_format == "NCHW":
        for no in range(kn_out):
            for ni in range(NUM_PE):
                for c in range(k_c_by_word):
                    for h in range(kh):
                        for w in range(kw):
                            idx_dst = h * (kw * kn_out * k_c_by_word * NUM_PE)
                            idx_dst += w * (kn_out * k_c_by_word * NUM_PE)
                            idx_dst += no * (k_c_by_word * NUM_PE)
                            idx_dst += c * (NUM_PE)
                            idx_dst += ni
                            transpose_values[idx_dst] = copy_value[idx_src]
                            idx_src += 1
    else:
        NotImplementedError("only NCHW and NHWC formats are suppported")

    return transpose_values

class NHWC_Transposer(GraphRunner):
    """Transposer of all nodes to NHWC."""

    def _get_permutation(self, dim: str) -> List[int]:
        """Create a permutation from the source dimension."""
        assert len(dim) == 4 and set(dim).issubset({'N', 'H', 'W', 'C', 'I', 'O'}), \
            f'illegal dimension found: {dim}'

        if set(dim) == set('HWIO'):
            dim = dim.replace('I', 'C')
            dim = dim.replace('O', 'N')

        return list(map(lambda s: dim.index(s), 'NHWC'))

    def _check_and_transpose(self, node: Operator) -> None:
        perm = self._get_permutation(node.dimension)
        node.transpose(perm)

    def run_backward_input(self, node: Input, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_constant(self, node: Constant, **kwargs: Any) -> None:
        if node.ndims == 4 and set(node.dimension).issubset({'N', 'H', 'W', 'C', 'I', 'O'}):
            self._check_and_transpose(node)

    def run_backward_identity(self, node: Identity, **kwargs: Any) -> None:
        if node.ndims == 4 and set(node.dimension).issubset({'N', 'H', 'W', 'C', 'I', 'O'}):
            self._check_and_transpose(node)

    def run_backward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_transpose(self, node: Transpose, **kwargs: Any) -> None:
        raise NotImplementedError('Transposing Transpose operator is not supported yet.')

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_batch_normalization(self, node: BatchNormalization, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_max_pool(self, node: MaxPool, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_average_pool(self, node: AveragePool, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_SpaceToDepth(self, node: SpaceToDepth, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_QTZ_binary_channel_wise_mean_scaling(
            self,
            node: QTZ_binary_channel_wise_mean_scaling,
            **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_ConcatOnDepth(self, node: ConcatOnDepth, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_Maximum(self, node: Maximum, **kwargs: Any) -> None:
        self._check_and_transpose(node)

    def run_backward_DepthToSpace(self, node: DepthToSpace, **kwargs: Any) -> None:
        self._check_and_transpose(node)


class PreComputeRunner(GraphRunner):
    """Optimization class that does precomputation and pruning on the graph.

    Fron a constant node, this object precomputes as far as possible, and
    replaces all precomputed nodes with a newly defined constant node.

    Additionally, in the hard-quantized mode, this object replaces a
    weight-quantizer node and succesive Conv node with a QConv node, and
    packs the weight.
    """

    _quantized_bitwidth = 1
    _wordsize = 32

    def __init__(self, graph: Graph, hard_quantized: bool = False) -> None:
        """Set up internal varibles."""
        self._precomp_dic: Dict[str, bool] = {}
        self._nodes_removed: Set[Operator] = set()
        self._hard_quantized = hard_quantized
        self._quantizers: Dict[str, Quantizer] = {}  # the operator name and its quantizer
        self._connected_convs: Dict[Operator, List[Conv]] = {}  # node name and its connected convolver

        super().__init__(graph)

    def initialize(self, **kwargs: Any) -> None:
        qconvs: List[Conv] = kwargs['qconv']
        self._connected_convs = {q: [q] for q in qconvs}

    def finalize(self, **kwargs: Any) -> None:
        """Remove all unused nodes from the graph."""
        for n in self._nodes_removed:
            self._graph.remove_op(n)

    # 1st phase: check which conv the node connects

    def run_backward_by_default(self, node: Operator, **kwargs: Any) -> None:
        outputs = node.output_op_list

        convs: List[Conv] = sum([self._connected_convs[out] for out in outputs if self._connected_convs.get(out)], [])
        self._connected_convs[node] = convs

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        pass  # do nothing, as all (quantized) conv node is already registered to self._connected_convs

    # 2nd phase: precompute and prune

    def _has_precompute_value(self, op: Operator) -> bool:
        """Return True if the operator has precompute value."""
        return self._precomp_dic[op.name]

    def _is_prunable(self, op: Operator) -> bool:
        """Return True if op can be prunable."""
        return self._has_precompute_value(op) and op.op_type != 'Constant'

    def _prune(self, node: Operator) -> None:
        """Prune the node and its inputs."""
        # prune inputs
        for i in node.input_ops.values():
            if i not in self._nodes_removed:
                self._prune(i)

        # prune itself
        self._nodes_removed.add(node)

    def _precompute_or_prune_inputs(self, node: Operator) -> None:
        """Precompute itself or prune the input nodes.

        If all input has precompute value, then make the node precompute.
        Otherwise, all prunable input nodes are pruned and substituted with
        a new constant node.
        """
        ops: List[Operator] = [node.input_ops[i] for i in node.input_names if node.input_ops.get(i)]
        ops_have_precomp_values = list(map(lambda x: self._has_precompute_value(x), ops))
        ops_are_prunable = list(map(lambda x: self._is_prunable(x), ops))
        ops_are_in_quantized = list(map(lambda x: x.name in self._quantizers.keys(), ops))

        # check which input node can be pruned
        if reduce(lambda x, y: x and y, ops_have_precomp_values):  # all input has concrete values
            node.run_forward()
            self._precomp_dic[node.name] = True  # this node can be pruned
            if reduce(lambda x, y: x or y, ops_are_in_quantized):  # some input operator to be quantized exists
                quantizers = {op.name: self._quantizers[op.name] for op in ops if self._quantizers.get(op.name)}
                if len(quantizers) > 1:
                    ValueError(f'{node.name}: multiple quantized inputs with {node.op_type} are not supported.')
                self._quantizers[node.name] = list(quantizers.values())[0]

        else:
            self._precomp_dic[node.name] = False

            # prune input opetarots
            for key, op in zip(node.input_names, ops):
                if self._is_prunable(op):
                    # get scaling factor if it is to be quantized but not in hard quantization mode
                    scaling = 1 if self._quantizers.get(op.name) is None \
                        else self._quantizers[op.name].scaling_factor

                    extra_dims = tuple(np.ones((len(op.data.shape) - len(scaling.shape)), dtype=np.int32))
                    scaling = scaling.reshape(scaling.shape + extra_dims)

                    # creates new constant
                    new_op = Constant(
                        op.name + '_new',
                        op.dtype,
                        op.data * scaling,
                        dimension_format=op.dimension
                    )

                    # replace and prune the old operators
                    node.add_input(key, new_op)
                    self._graph.add_op(new_op)
                    self._prune(op)

    def run_forward_by_default(self, node: Operator, **kwargs: Any) -> None:
        self._precompute_or_prune_inputs(node)

    def run_forward_input(self, node: Input, **kwargs: Any) -> None:
        self._precomp_dic[node.name] = False

    def run_forward_constant(self, node: Constant, **kwargs: Any) -> None:
        self._precomp_dic[node.name] = True

    def run_forward_identity(self, node: Identity, **kwargs: Any) -> None:
        """skip all identity."""
        in_op = node.input_ops['input']
        out_ops = node.output_ops['output']
        for out_op in out_ops:
            for k, v in out_op.input_ops.items():
                if v == node:
                    # change the output's input to this identity's input
                    out_op.add_input(k, in_op)
                    # change the input's output to this identity's output
                    for k2, v2 in in_op.output_ops.items():
                        if node in v2:
                            v2.remove(node)
                            v2.append(out_op)
                            break
                    break

    def run_forward_QTZ_binary_mean_scaling(self, node: QTZ_binary_mean_scaling, **kwargs: Any) -> None:
        in_op = node.input_ops['input']

        # if it can be precomputed
        if self._has_precompute_value(in_op):
            node.run_forward()
            self._precomp_dic[node.name] = True  # this node can be pruned
            self._quantizers[node.name] = node  # add itself as the quantizer
        else:
            self._precomp_dic[node.name] = False

    def run_forward_conv(self, node: Conv, **kwargs: Any) -> None:
        ops: List[Operator] = [node.input_ops[i] for i in node.input_names if node.input_ops.get(i)]

        if self._hard_quantized and node in kwargs['qconv']:
            # data is to be packed
            ops_have_precomp_values = list(map(lambda x: self._has_precompute_value(x), ops))
            ops_are_prunable = list(map(lambda x: self._is_prunable(x), ops))

            # check which input node can be pruned
            if reduce(lambda x, y: x and y, ops_have_precomp_values):  # all input has concrete values
                node.run_forward()
                self._precomp_dic[node.name] = True  # this node can be pruned
                quantizers = {op.name: self._quantizers[op.name] for op in ops if self._quantizers.get(op.name)}
                if len(quantizers) > 1:
                    ValueError(f'{node.name}: multiple quantized inputs with {node.op_type} are not supported.')
                self._quantizers[node.name] = list(quantizers.values())[0]

            else:   # an input (must be weight) is to be quantized and packed
                self._precomp_dic[node.name] = False
                node.is_quantized = True
                packer = Packer(self._quantized_bitwidth, self._wordsize)
                quantizers = {op.name: self._quantizers[op.name] for op in ops if self._quantizers.get(op.name)}
                if len(quantizers) > 1:
                    ValueError(f'{node.name}: multiple quantized inputs with {node.op_type} are not supported.')
                node.quantizer = list(quantizers.values())[0]

                for key, op in zip(node.input_names, ops):

                    if self._is_prunable(op):
                        oh = node.height
                        ow = node.width
                        od = node.channel
                        kh = node.kernel_height
                        kw = node.kernel_width
                        kd = op.channel
                        shape = op.shape
                        op_data = node.quantizer.binarizer(op.data)
                        data = packer.run(op_data.astype(np.float32), op.dimension)
                        dtype = op.dtype
                        new_op = Constant(
                            op.name + '_new',
                            dtype,
                            data,
                            packed=True,
                            actual_shape=shape,
                            transposed_data=transpose_kernels(data, node.dimension, oh, ow, od, kh, kw, kd)
                        )
                        node.add_input(key, new_op)
                        self._graph.add_op(new_op)
                        self._prune(op)

        else:
            self._precompute_or_prune_inputs(node)

    def run_forward_QTZ_binary_channel_wise_mean_scaling(
            self,
            node: QTZ_binary_channel_wise_mean_scaling,
            **kwargs: Any) -> None:
        in_op = node.input_ops['input']

        # if it can be precomputed
        if self._has_precompute_value(in_op):
            node.run_forward()
            self._precomp_dic[node.name] = True  # this node can be pruned
            self._quantizers[node.name] = node  # add itself as the quantizer
        else:
            self._precomp_dic[node.name] = False


class DTypeChanger(GraphRunner):
    """Optimization class that changes dypes.

    This runner must run before PrecomputeRunner.
    """

    class Path(Enum):
        INPUT = 1,
        WEIGHT = 2,
        OTHER = 3

    _packed_dtype = {Path.INPUT: QUANTIZED_NOT_PACKED(), Path.WEIGHT: Uint32(), Path.OTHER: Float32()}
    _a_quantizers = {'QTZ_linear_mid_tread_half'}
    _w_quantizers = {'QTZ_binary_mean_scaling', 'QTZ_binary_channel_wise_mean_scaling'}
    _conv = {'Conv'}

    def __init__(self, graph: Graph) -> None:
        """Set up internal varibles."""
        self._output_convs: Dict[Operator, List[Conv]] = {}
        self._packed_input_path: Dict[str, Any] = {}

        super().__init__(graph, depth_first=False)

    # 1st phase: check nodes which dtype must be changed

    def _check_dtype_state(self, node: Operator) -> None:
        """checks the state of each node regarding dtype.

        - whether the node is after conv and before activation quantizer
        - whether the node is after activation and before conv
        """
        outputs = node.output_op_list
        convs: List[Conv] = sum([self._output_convs[out] for out in outputs if self._output_convs.get(out) is not None],
                                [])

        # determine the path of node is input or weight or others
        path = self.Path.WEIGHT
        for out in outputs:
            p = self._packed_input_path[out.name] if out.op_type not in self._conv \
                else self.Path.INPUT if node == out.input_ops['X'] \
                else self.Path.WEIGHT
            if path == self.Path.WEIGHT:
                path = p
            elif path == p:
                pass
            else:  # output have different paths
                ValueError('multiple outputs must have the same kind of paths.')

        is_not_before_a_quantizer = reduce(lambda x, y: x and y,
                                           [out.op_type not in self._a_quantizers for out in outputs])
        if convs and is_not_before_a_quantizer:
            self._output_convs[node] = convs

        self._packed_input_path[node.name] = path

    def run_backward_by_default(self, node: Operator, **kwargs: Any) -> None:
        self._check_dtype_state(node)

    def run_backward_output(self, node: Output, **kwargs: Any) -> None:
        self._packed_input_path[node.name] = self.Path.OTHER

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        self._output_convs[node] = [node]

    # 2nd phase: change data type

    def turn(self, **kwargs: Any) -> None:
        """Set up qconv list"""
        output_convs: List[Conv] = sum(list(self._output_convs.values()), [])
        for conv in output_convs:
            # get all ascendants of conv
            ascendants = [k for k in self._output_convs.keys() if conv in self._output_convs[k]]

            # whether some weight quantizer is in ascendants
            wqtz_in_asc = reduce(lambda x, y: x or y,
                                 list(map(lambda n: n.op_type in self._w_quantizers, ascendants)))
            # whether some activation quantizer is in ascendants
            aqtz_in_asc = reduce(lambda x, y: x or y,
                                 list(map(lambda n: n.op_type in self._a_quantizers, ascendants)))
            # if both, add conv to the list
            if wqtz_in_asc and aqtz_in_asc:
                kwargs['qconv'].add(conv)

    def _set_dtype(self, node: Operator, qconv: List[Conv]) -> None:
        def before_qconv() -> bool:
            """Return if the node is before a quantized convolver"""
            convs: List[Conv] = self._output_convs[node] if self._output_convs.get(node) else []
            # consistency check
            is_qconv: List[bool] = list(map(lambda x: x in qconv, convs))
            all_is_qconv = reduce(lambda x, y: x and y, is_qconv, True)
            some_is_qconv = reduce(lambda x, y: x or y, is_qconv, False)
            assert convs == [] or (all_is_qconv == some_is_qconv), \
                f'{node.name} connects to both of a quantized convolver and non-quantized one.'

            return convs != [] and all_is_qconv

        def get_dtype() -> Optional[DataType]:
            """Return dtype along with which path the node is on: 'input' or 'weight' of a conv"""
            path = self._packed_input_path.get(node.name)
            return self._packed_dtype[path] if path is not None else None

        dtype = get_dtype()
        conv = self._output_convs.get(node)
        if dtype is not None and before_qconv():
            node.dtype = dtype

    def run_forward_by_default(self, node: Operator, **kwargs: Any) -> None:
        self._set_dtype(node, kwargs['qconv'])


class ApplyThresholdSkipping(GraphRunner):
    """Optimization class that perform threshold skipping.

    This runner perform threshold skipping with BFS for DLK graph.
    Run graphrunner backward to acquire graph info, and run forward
    to compute the thresholds skip batchnorm and activation quantizer
    with thresholding function.
    """

    def __init__(self, graph: Graph) -> None:
        self._aqtz_aqtz: Dict[Operator, List[Operator]] = {}
        self._qconv_qconv: Dict[Conv, List] = {}
        super().__init__(graph, depth_first=False)

    def _apply_threshold_skipping(self, op_lst: List[Operator]) -> None:
        """Performs Conv thresholds computation and skipping."""

        transitions: Dict[int, Operator] = {}
        start, finish = [None, None]
        for idx, op in enumerate(op_lst):
            if node_is_qconv(op):
                start = cast(Conv, op)
            elif node_is_activation_quantizer(op):
                finish = op
                transitions[idx] = op
            else:
                transitions[idx] = op

        if start is not None and finish is not None:

            def linear_qtz2float(x: np.ndarray, n_value: int, max_value: float) -> np.ndarray:
                real_x = x / np.float64(n_value) * np.float64(max_value)
                return real_x.astype(np.float64)

            # Step 1: Compute thresholds for Convolution operators
            aqtzer = cast(Quantizer, start.a_quantizer[0])  # Activation Quantizers should all have the same bits
            bit = aqtzer.nbit
            max_v = aqtzer.max_v
            if bit is None or max_v is None:
                ValueError(f'activation quantizer of node {start.name} has bit or max value of None')

            n = 2 ** bit - 1
            ch = start.channel
            lch = start.input_ops['X'].channel
            k = start.kernel_height * start.kernel_width * lch * n
            qtzer = cast(Quantizer, start.quantizer)
            conv_results = [x for x in range(-k, k + 1, 1)]
            th_tmp = np.empty([ch, n + 1], dtype=np.int32)
            v_now = dict.fromkeys([x for x in range(ch)], 0)
            th_now = 0
            val_neg_flag = -1
            val_pos_flag = 1
            all_transdata: Dict[int, Dict[str, Any]] = {}

            # Step 1-1: initalize thresholds
            for conv_res in conv_results:
                conv_out = np.full(ch, conv_res, dtype=np.float64)
                conv_out *= qtzer.scaling_factor if qtzer.scaling_factor is not None \
                    else ValueError(f'oops Quantizer of node {start.name} has scaling factor of None')

                conv_data = linear_qtz2float(conv_out, n, max_v)

                trans_data: Dict[str, Any] = {'data': conv_data}
                for idx, op in sorted(transitions.items(), reverse=True):
                    trans_data = op.run(**trans_data)

                for depth in range(ch):
                    init = -k if depth in trans_data['nega_idx'] else k
                    th_tmp[depth, :] = init

                all_transdata[conv_res] = trans_data

            # Step 1-2: update thresholds
            for conv_res in conv_results:
                trans_data = all_transdata[conv_res]
                qtz_out = trans_data['data']
                qtz_mu = np.mean(qtz_out)
                if qtz_mu != th_now:
                    for depth in range(ch):
                        is_negative = depth in trans_data['nega_idx']
                        if v_now.get(depth) != qtz_out[depth]:
                            if is_negative:
                                th_tmp[depth, abs(n - qtz_out[depth] - 1)] = conv_res
                            else:
                                th_tmp[depth, qtz_out[depth] - 1] = conv_res
                            v_now[depth] = qtz_out[depth]
                        th_tmp[depth, n] = -1 if is_negative else 1
                for depth in range(ch):
                    constant = reduce(lambda x, y: x and y,
                                      [th_tmp[depth, i] == th_tmp[depth, i + 1] for i in range(n - 1)])
                    th_tmp[depth, n] = qtz_out[depth] + 2 if constant else th_tmp[depth, n]
                    # note: 2 above is a magic number. the result value must not be 1 nor -1.
                th_now = qtz_mu

            start.thresholds = th_tmp.flatten().tolist()

            # Step 2: Skipping unused operators, e.g. batch normalization, linear activation quantizer
            if start.has_thresholds:
                if start.dtype is not finish.dtype:
                    start.dtype = finish.dtype
                for consumers in finish.output_ops.values():
                    for consumer in consumers:
                        for idex, y in start.output_ops.items():
                            if not bool(set(consumers) & set(y)):
                                start.remove_output(idex)
                            start.add_output(idex, consumer)

                        for indent, v in consumer.input_ops.items():
                            if v == finish:
                                consumer.add_input(indent, start)
                                break
        else:
            pass

    def _makeup_skippable(self, node: Operator) -> None:
        outputs = node.output_op_list
        for out_op in outputs:
            for start, lst in self._aqtz_aqtz.items():
                if out_op in lst:
                    self._aqtz_aqtz[start].append(node)

    def _makeup_aqtz(self, node: Operator) -> None:
        outputs = node.output_op_list
        for out_op in outputs:
            for start, lst in self._qconv_qconv.items():
                if out_op in lst:
                    self._qconv_qconv[start].append(node)

    def run_backward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        self._aqtz_aqtz[node] = [node]
        self._makeup_aqtz(node)

    def run_backward_by_default(self, node: Operator, **kwargs: Any) -> None:
        if node.is_monotonic and not node_is_conv(node):
            self._makeup_skippable(node)
        self._makeup_aqtz(node)

    def run_backward_conv(self, node: Conv, **kwargs: Any) -> None:
        self._makeup_skippable(node)
        if node_is_qconv(node):
            self._qconv_qconv[node] = [node]

    def run_forward_conv(self, node: Conv, **kwargs: Any) -> None:
        bits: List[int] = []
        aqtzers: List[Quantizer] = []
        if node_is_qconv(node):
            for x in self._qconv_qconv[node]:
                if node_is_activation_quantizer(x):
                    bits.append(x.nbit)
                    aqtzers.append(x)

        if not (len(set(bits)) == 1):
            ValueError('Values are not consistent')
        else:
            node.a_quantizer = aqtzers

    def run_forward_QTZ_linear_mid_tread_half(self, node: QTZ_linear_mid_tread_half, **kwargs: Any) -> None:
        self._apply_threshold_skipping(self._aqtz_aqtz[node])


class Optimizer(object):
    """Class of optimization classes."""

    def transpose_NHWC(self, graph: Graph) -> Graph:
        runner = NHWC_Transposer(graph)
        kwargs: Dict[str, Any] = {}
        runner.run(**kwargs)
        return graph

    def precompute(self, graph: Graph, hard_quantized: bool = False) -> Graph:
        runner1 = DTypeChanger(graph)
        runner2 = PreComputeRunner(graph, hard_quantized=hard_quantized)

        kwargs: Dict[str, Set[Conv]] = {'qconv': set()}

        # run
        if hard_quantized:
            runner1.run(**kwargs)
        runner2.run(**kwargs)

        return graph

    def threshold_skipping(self, graph: Graph) -> Graph:
        runner1 = ApplyThresholdSkipping(graph)
        kwargs: Dict[str, Any] = {}
        runner1.run(**kwargs)
        return graph

