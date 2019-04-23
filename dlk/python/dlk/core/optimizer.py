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
import math
import warnings
import numpy as np

from core.graph import Graph
from core.graph_pattern_matching import get_nodes_in_branch, sort_graph
from core.operators import Constant, Operator, Conv
from core.data_types import PackedUint32, QUANTIZED_NOT_PACKED
from typing import cast, List, Any
from collections import defaultdict
from modules.packer import Packer


def _transpose_kernels(kernel_data: np.ndarray,
                       oh: int,
                       ow: int,
                       od: int,
                       kh: int,
                       kw: int,
                       kd: int) -> List[int]:
    """Calculates and prepares the transposed kernel data in advance.

    Parameters
    ----------
    kernel_data : np.ndarray
        The input data.
    oh : int
        output height
    ow : int
        output width
    od : int
        output depth
    kh : int
        kernel height
    kw : int
        kernel width
    kd : int
        kernel depth
    """
    NUM_PE = 16
    NBIT_QDYPE = 32
    MAX_NBIT_QINPUT = 2
    MAX_NBIT_KERNEL = 1
    num_qinput_per_qword = int(NBIT_QDYPE / MAX_NBIT_QINPUT)
    num_qkernel_per_qword = int(NBIT_QDYPE / MAX_NBIT_KERNEL)
    k_c_by_word = int((kd + (num_qkernel_per_qword - 1)) / num_qkernel_per_qword)
    k_n_aligned_with_num_pe = int(((od + (NUM_PE - 1)) / NUM_PE) * NUM_PE)
    if od < NUM_PE:
        k_size = k_n_aligned_with_num_pe * kh * kw * k_c_by_word
    else:
        k_size = od * kh * kw * k_c_by_word

    flatten_value = []
    for elem in kernel_data:
        flatten_value.extend(elem)
    while len(flatten_value) != k_size:
        flatten_value.extend("0")

    copy_value = [0] * k_size
    for i in range(od * kh * kw * k_c_by_word):
        copy_value[i] = flatten_value[i]

    transposed_values = [0] * k_size
    if (od < NUM_PE):
        kn_out = int(k_n_aligned_with_num_pe / NUM_PE)
    else:
        kn_out = int(od / NUM_PE)
    idx_src = 0

    for no in range(kn_out):
        for ni in range(NUM_PE):
            for h in range(kh):
                for w in range(kw):
                    for c in range(k_c_by_word):
                        idx_dst = no * (kh * kw * k_c_by_word * NUM_PE)
                        idx_dst += h * (kw * k_c_by_word * NUM_PE)
                        idx_dst += w * (k_c_by_word * NUM_PE)
                        idx_dst += c * (NUM_PE)
                        idx_dst += ni
                        transposed_values[idx_dst] = copy_value[idx_src]
                        idx_src += 1

    return transposed_values


def pass_remove_identities(graph: Graph) -> None:
    """Removes those nodes of a Graph that satisfies the condition node.op_type() == Identity.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.

    """
    exec_list = [n for n in sort_graph(graph) if n.op_type == 'Identity']
    to_be_removed = list()
    for m in exec_list:
        """skip all identity."""
        in_op = m.input_ops['input']
        out_ops = m.output_ops['output']
        for out_op in out_ops:
            for k, v in out_op.input_ops.items():
                if v == m:
                    # change the output's input to this identity's input
                    out_op.add_input(k, in_op)
                    # change the input's output to this identity's output
                    for k2, v2 in in_op.output_ops.items():
                        if m in v2:
                            v2.remove(m)
                            v2.append(out_op)
                            break
                    break

        to_be_removed.append(m)

    for op in to_be_removed:
        graph.remove_op(op)


def pass_transpose(graph: Graph) -> None:
    """Changes the data order of every node to be NHWC.
       The fastest changing dimension is C
       N stands for batch size (on inference we assume is 1.
       H and W are the height and width respectively.
       C stands for depth (aka channels)

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.

    """
    exec_list = sort_graph(graph)

    for m in exec_list:
        dim = m.dimension
        shape = m.shape
        if len(shape) != 4 or len(dim) != 4 or not set(dim).issubset({'N', 'H', 'W', 'C', 'I', 'O'}):
            continue

        dim = dim.replace('I', 'C')
        dim = dim.replace('O', 'N')

        permutation = list(map(lambda s: dim.index(s), 'NHWC'))
        m.transpose(permutation)


def pass_constant_folding(graph: Graph) -> None:
    """Given a node N, if the value of each input of N is known at compilation time then N will be executed.
       The node N and its inputs will be replaced with a Constant node which holds the computed output of N.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    processed_nodes : list
        The list of the processed nodes so far.
    """

    done = False
    processed_nodes = []
    while not done:
        exec_list = sort_graph(graph)
        processed_before_precompute = len(processed_nodes)
        to_be_removed = []

        for m in exec_list:
            if m in processed_nodes:
                continue

            # We want operators with inputs
            if not m.input_nodes:
                continue

            precomputable = True
            for input_node in m.input_nodes:
                if input_node.op_type != 'Constant':
                    precomputable = False

            if not precomputable:
                continue

            processed_nodes += m.input_nodes
            processed_nodes.append(m)

            data = m.run_forward()

            new_constant = Constant(
                m.name + '_new',
                m.dtype,
                data,
                dimension_format=m.dimension
            )
            graph.add_op(new_constant)

            # get nodes to be removed after being disconnected
            get_nodes_in_branch(m, None, to_be_removed)

            new_constant.add_outputs({'output': m.output_ops.values()})
            for output_name, consumer_list in m.output_ops.items():
                for consumer_node in consumer_list:
                    for input_name, input_node in consumer_node.input_ops.items():
                        if input_node == m:
                            consumer_node.add_input(input_name, new_constant)
                            break

        for op in to_be_removed:
            graph.remove_op(op)

        done = len(processed_nodes) == processed_before_precompute


def pass_propagate_quantization_details_into_conv(graph: Graph) -> None:
    """Given a node N, it will propagate information about quantization into the convolution nodes.

       There are two types of nodes. Those which preserve quantization (for example, Space2Depth because
       does not affect the actual values of the input data, only changes it positions) and those which
       destroy quantization (for example, BatchNormalization, because it involves float operations).

       If there is path in the Graph which connect a Quantizer node Q to a Conv node C and every node between
       Q and C preserve quantization (for example, Q -> Space2Depth -> Concat > Conv) then the details about the
       quantizer Q should be propagated into the convolution node C.

       This pass allows us to further process the convolution nodes later and maybe quantize these convolutions
       based on these quantization details. Note that a convolution node has two inputs, input data and weights.
       We propagate quantization details through both the input node branch and the weight node branch.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    """

    exec_list = sort_graph(graph)
    qtypes = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'QTZ_binary_channel_wise_mean_scaling'
    ]

    quant_details = defaultdict(list)
    for m in exec_list:
        if not m.preserve_quantization:
            quant_details[m.name] = []
            continue

        if m.op_type == 'Conv':
            input_node = m.input_nodes[0]
            weight_node = m.input_nodes[1]

            m.a_quantizer = [input_node] if input_node.op_type in qtypes else quant_details[input_node.name]
            m.quantizer = weight_node if weight_node.op_type in qtypes else quant_details[weight_node.name]

            quant_details[m.name] = []
        else:
            qtzs = []
            for n in m.input_nodes:
                if n.op_type in qtypes:
                    qtzs.append(n)
                else:
                    for q in quant_details[n.name]:
                        qtzs.append(q)

            if qtzs:
                nbits = []
                max_vs = []
                for qtz in qtzs:
                    nbits.append(qtz.nbit)
                    max_vs.append(qtz.max_v)
                if not (len(set(nbits)) == 1) and not (len(set(max_vs)) == 1):
                    warnings.warn(f'bits {nbits} or max values {max_vs} are not consistent '
                                  f'to propagate quantization information to {m.name}')
                    quant_details[m.name] = []
                else:
                    quant_details[m.name] = qtzs
            else:
                quant_details[m.name] = []


def pass_compute_thresholds(graph: Graph) -> None:
    """Given a Quantizer node Q:
         - if there is a backward path between Q and a convolution node and,
         - every node N of that path satisfies the condition N.is_monotonic and,
         - the convolution node C (the end of this path) is a quantized convolution
       then this pass construct an LUT per channel which maps a possible output value of the quantized convolution node
       C to the corresponding output of the quantization node Q. This effectively compress the path C -> ... -> Q
       into a list of LUTs that can be used during inference.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    """
    exec_list = [n for n in sort_graph(graph) if n.op_type == 'QTZ_linear_mid_tread_half']
    to_be_removed = []
    for m in exec_list:
        # find a a backward path between the quantizer and the convolution ie. a path represented by a list [Q, ..., C]
        p = [m]
        while p[-1].op_type != 'Conv':
            non_variable_input = [inode for inode in p[-1].input_nodes
                                  if (not cast(Operator, inode).is_variable and inode.is_monotonic)
                                  or inode.op_type == 'Conv']
            if len(non_variable_input) != 1:
                break
            p.append(non_variable_input[-1])

        if p[-1].op_type != 'Conv':
            continue
        activation_quantizer_node = p[0]
        conv_node = p[-1]

        # check if this is a quantized convolution
        if not conv_node.quantizer or not conv_node.a_quantizer:
            continue

        quantizer_conv_weights = conv_node.quantizer
        quantizer_conv_weights.run_forward_no_scaling_factor()
        scaling_factor = quantizer_conv_weights.scaling_factor

        # Getting the bit and max value
        nbits = []
        max_vs = []
        for aqtz in conv_node.a_quantizer:
            nbits.append(aqtz.nbit)
            max_vs.append(aqtz.max_v)
        if not (len(set(nbits)) == 1) and not (len(set(max_vs)) == 1):
            raise ValueError(f'bits {nbits} or max values {max_vs} are not consistent')
        else:
            nbit = nbits[0]
            max_v = max_vs[0]

        n = 2 ** nbit - 1
        ch = conv_node.channel
        # assume that the threshold values will be a 13-bit signed integer
        max_th_value = 2 ** 12 - 1

        # The threshold_table is numpy array that holds the threshold values for all channels
        threshold_table = np.empty([ch, n + 1], dtype=np.int32)

        # Compute threshold (t0, t1, t2)
        th_val = [0.5 + i for i in range(n)]
        for th_id, th_v in enumerate(th_val):
            init_threshold = np.full(ch, th_v, dtype=np.float64)

            # run calculation in reverse order, for example, q -> bn -> scaling
            bn_nega_idx = []
            trans_th = {'data': init_threshold}
            for op in p[:-1]:
                trans_th = op.de_run(**trans_th)
                if op.op_type == 'BatchNormalization':
                    bn_scale = op.input_ops['scale'].data
                    bn_nega_idx = [v for v in range(len(bn_scale)) if bn_scale[v] < 0]
            threshold = (trans_th['data'] * np.float64(n)) / (np.float64(max_v) * scaling_factor)

            # take care of threshold values that are larger than 13-bit signed integer
            threshold[threshold > max_th_value] = max_th_value
            threshold[threshold < -max_th_value] = -max_th_value

            for ch_id, th_per_ch in enumerate(threshold):
                if quantizer_conv_weights.op_type == 'QTZ_binary_channel_wise_mean_scaling':
                    threshold_table[ch_id, th_id] = int(math.floor(th_per_ch)) \
                        if (scaling_factor[ch_id] < 0) ^ (ch_id in bn_nega_idx) \
                        else int(math.ceil(th_per_ch))
                else:
                    threshold_table[ch_id, th_id] = int(math.floor(th_per_ch)) \
                        if (scaling_factor < 0) ^ (ch_id in bn_nega_idx) \
                        else int(math.ceil(th_per_ch))

        for c in range(ch):
            threshold_table[c, -1] = 1 \
                if np.all(threshold_table[c, 1:-1] > threshold_table[c, :-2], axis=0) else -1
            # Applying the magic number
            if np.all(threshold_table[c, 1:-1] == threshold_table[c, :-2], axis=0):
                threshold_table[c, -1] = 2

        # Put the thresholds into list
        conv_node.thresholds = threshold_table.flatten().tolist()

        # get nodes to be removed after being disconnected
        get_nodes_in_branch(activation_quantizer_node, conv_node, to_be_removed)

        # Disconnect the outputs of the quantizer
        out_ops = activation_quantizer_node.output_ops['output']
        for output_node in out_ops:
            for input_name, input_node in output_node.input_ops.items():
                if input_node == activation_quantizer_node:
                    output_node.add_input(input_name, conv_node)

        # Disconnect the outputs of the conv
        conv_node.remove_output('Y')
        conv_node.add_outputs({'Y': out_ops})

    for op in to_be_removed:
        graph.remove_op(op)


def pass_pack_weights(graph: Graph) -> None:
    """Given a Quantized convolution node C, it will pack the weights of C into 32 bit words.
       If the node Q that apply quantization to the weights of C quantizes, for example, into 1 bit values
       then one 32 bit word will contain 32 weights.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    """
    exec_list = [n for n in sort_graph(graph) if n.op_type == 'Conv']
    quantization_types = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'QTZ_binary_channel_wise_mean_scaling'
    ]

    word_size = 32
    weight_bitwidth = 1
    packer = Packer(weight_bitwidth, word_size)
    to_be_removed = []

    for m in exec_list:
        conv_node = m

        # check if this is a quantized convolution
        if not conv_node.quantizer or not conv_node.a_quantizer:
            continue

        # Check if we support this kind of quantizer
        weight_quantizer = conv_node.quantizer
        if weight_quantizer.op_type not in quantization_types:
            continue

        # Quantize the weights
        weight_quantizer.run_forward()
        op_data = weight_quantizer.binarizer(weight_quantizer.data)
        data = packer.run(op_data.astype(np.float32), weight_quantizer.dimension)

        # Create the new constant with the quantized weights
        oh = conv_node.height
        ow = conv_node.width
        od = conv_node.channel
        kh = conv_node.kernel_height
        kw = conv_node.kernel_width
        kd = conv_node.input_ops['X'].channel
        quantized_constant = Constant(
            weight_quantizer.name + '_new',
            PackedUint32(),
            data,
            packed=True,
            actual_shape=weight_quantizer.shape,
            transposed_data=_transpose_kernels(data, oh, ow, od, kh, kw, kd)
        )

        # get nodes to be removed after being disconnected
        get_nodes_in_branch(weight_quantizer, None, to_be_removed)

        # Add the constant to the graph and connect the new constant
        graph.add_op(quantized_constant)
        quantized_constant.add_outputs(weight_quantizer.output_ops)
        for output_name, consumer_list in weight_quantizer.output_ops.items():
            for consumer_node in consumer_list:
                for input_name, input_node in consumer_node.input_ops.items():
                    if input_node == weight_quantizer:
                        consumer_node.add_input(input_name, quantized_constant)
                        break

    for op in to_be_removed:
        graph.remove_op(op)


def pass_quantize_convolutions(graph: Graph) -> None:
    """Given a convolution node C, if C has proper quantization details, it will mark C as quantized and it will
       assign the correct output data types to the node C and its quantizers. Note that the expected output data type
       on the runtime is defined as QUANTIZED_NOT_PACKED.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    """
    exec_list = [n for n in sort_graph(graph) if n.op_type == 'Conv']
    for m in exec_list:
        conv_node = m

        # check if this is a quantized convolution
        if not conv_node.quantizer or not conv_node.a_quantizer:
            continue

        # Mark as quantized convolution
        conv_node.is_quantized = True

        # change the output data type of the convolution if thresholds are available
        if conv_node.has_thresholds:
            conv_node.dtype = QUANTIZED_NOT_PACKED()

        # change the output data type of the quantizers
        conv_node.quantizer.dtype = PackedUint32()
        for qtz in conv_node.a_quantizer:
            qtz.dtype = QUANTIZED_NOT_PACKED()


def pass_propagate_datatypes(graph) -> None:
    """Further propagate output data types.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    """
    exec_list = sort_graph(graph)
    for m in exec_list:
        if m.op_type != 'Conv' and m.preserve_quantization:
            m.dtype = m.input_nodes[0].dtype


def pass_propagate_output_type_backward(graph: Graph) -> None:
    """It is assumed that the output data type of a Graph is float.
       We should propagate this assumption backwards from the output node of the graph to the
       latest quantized convolution available.

       There could be cases where the latest convolution node Q is a quantized convolution and we also apply
       thresholds to its outputs. In this cases, the quantized convolution output data type should be float
       even if thresholds are applied.

    Parameters
    ----------
    graph : Graph
        The input graph. It will be modified in-place.
    """
    exec_list = sort_graph(graph)

    def output_dtype_changer(node, otype):
        for n in node.input_nodes:
            if n.op_type == 'Conv' and n.is_quantized:
                n.dtype = otype
                return
            output_dtype_changer(n, otype)

    # propagate output data type to the last quantized convolution
    output_node = exec_list[-1]
    output_type = output_node.dtype
    output_dtype_changer(output_node, output_type)
