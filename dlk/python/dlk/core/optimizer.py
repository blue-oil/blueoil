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
from collections import defaultdict
from typing import Any, List, cast

import numpy as np

from core.data_types import QUANTIZED_NOT_PACKED, QUANTIZED_PACKED, QUANTIZED_PACKED_KERNEL, Int32, PackedUint32, Uint32
from core.graph import Graph
from core.graph_pattern_matching import get_nodes_in_branch, sort_graph
from core.operators import Constant, Conv, Lookup, Operator, BatchNormalizationOptimized
from modules.packer import Packer


def pass_remove_identities(graph: Graph) -> None:
    """Removes those nodes of a Graph that satisfies the condition node.op_type() == Identity.

    Args:
        graph (Graph): The input graph. It will be modified in-place.
    
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

    Args:
        graph (Graph): The input graph. It will be modified in-place.
    
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

    Args:
        graph (Graph): The input graph. It will be modified in-place.
        processed_nodes (list): The list of the processed nodes so far.
    
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

    Args:
        graph (Graph): The input graph. It will be modified in-place.

    """

    exec_list = sort_graph(graph)
    qtypes = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'BinaryChannelWiseMeanScalingQuantizer',
        'Lookup'
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

    Args:
        graph (Graph): The input graph. It will be modified in-place.
    
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
                if quantizer_conv_weights.op_type == 'BinaryChannelWiseMeanScalingQuantizer':
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
            if np.all(threshold_table[c, 1:-1] == threshold_table[c, :-2], axis=0):
                threshold_table[c, -1] = 1
                threshold_table[c, 0:-1] = max_th_value

        bits_per_word = 32
        rem = (bits_per_word - ch % bits_per_word) % bits_per_word
        pad = np.ones((rem, n + 1), dtype=np.int32)
        threshold_table = np.vstack((threshold_table, pad))

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

    Args:
        graph (Graph): The input graph. It will be modified in-place.
    
    """
    exec_list = [n for n in sort_graph(graph) if n.op_type == 'Conv']
    quantization_types = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'BinaryChannelWiseMeanScalingQuantizer'
    ]

    word_size = 32
    weight_bitwidth = 1
    packer = Packer(weight_bitwidth, word_size)
    to_be_removed = []
    b = 32

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

        def pad_to_multiple_of_b(tensor, axis, b):
            shape = list(tensor.shape)
            pad = (((shape[axis] + b - 1) // b) * b) - shape[axis]
            shape[axis] = pad
            return np.zeros(shape) if pad else None

        padded_data = np.copy(weight_quantizer.data)

        for axis in [0, 3]:
            pad_tensor = pad_to_multiple_of_b(padded_data, axis, b)
            if pad_tensor is not None:
                padded_data = np.append(padded_data, pad_tensor, axis=axis)

        tca_output = np.copy(padded_data)
        oc, kh, kw, kd = padded_data.shape[:]
        padded_data = padded_data.flatten()
        tca_output = tca_output.flatten()

        out_index = 0
        for g in range(oc // b):
            for p in range(kd // b):
                for h in range(kh):
                    for w in range(kw):
                        for o in range(b):
                            for d in range(b):
                                idx = g * (kw * kh * kd * b) + p * b + h * (kw * kd) + w * kd + o * (kw * kh * kd) + d
                                tca_output[out_index] = padded_data[idx]
                                out_index += 1
                                
        kn2row_output = np.zeros(oc * kh * kw * kd)
        out_index = 0
        for h in range(kh):
            for w in range(kw):
                for o in range(oc):
                    for i in range(kd):
                        idx = o * kh * kw * kd + h * kw * kd + w * kd + i
                        kn2row_output[out_index] = padded_data[idx]
                        out_index += 1

        op_data = weight_quantizer.binarizer(padded_data)
        data = packer.run(op_data.astype(np.float32), weight_quantizer.dimension)

        tca_binarized_data = weight_quantizer.binarizer(tca_output)
        tca_packed_data = packer.run(tca_binarized_data.astype(np.float32), weight_quantizer.dimension)

        kn2row_binarized_data = weight_quantizer.binarizer(kn2row_output)
        kn2row_data = packer.run(kn2row_binarized_data.astype(np.float32), weight_quantizer.dimension)

        shape = [oc, kh, kw, kd]
        tca_shape = [oc // b, kd // b, kh, kw, b, b]
        kn2row_shape = [kh, kw, oc, kd]

        # Create the new constant with the quantized weights
        quantized_constant = Constant(
            weight_quantizer.name + '_new',
            PackedUint32(),
            data=np.vectorize(lambda k: (~k) & ((0x1 << 32) - 1))(data),
            dimension_format="OHWI",
            transposed_dimension_format="OhIhHWOlIl",
            packed=True,
            actual_shape=shape,
            transposed_shape=tca_shape,
            transposed_data=[(~k) & ((0x1 << 32) - 1) for k in tca_packed_data.flatten()],
            kn2row_data=[k for k in kn2row_data.flatten()],
            kn2row_shape=kn2row_shape,
            kn2row_dimension_format="HWOI"
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

    Args:
        graph (Graph): The input graph. It will be modified in-place.
    
    """
    b = 32

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
            conv_node.dtype = QUANTIZED_PACKED()
            height = conv_node.height
            width = conv_node.width
            depth = conv_node.channel
            depth_upper = (depth + b - 1) // b
            conv_node.update_shape([depth_upper, height, width, 2, b], "ChHWBCl")

        # change the output data type of the quantizers
        conv_node.quantizer.dtype = PackedUint32()
        for qtz in conv_node.a_quantizer:
            if isinstance(qtz, Lookup):
                continue
            qtz.dtype = QUANTIZED_PACKED()
            height = qtz.height
            width = qtz.width
            depth = qtz.channel
            depth_upper = (depth + b - 1) // b
            qtz.update_shape([depth_upper, height, width, 2, b], "ChHWBCl")


def pass_propagate_datatypes(graph) -> None:
    """Further propagate output data types.

    Args:
        graph (Graph): The input graph. It will be modified in-place.
    
    """
    exec_list = sort_graph(graph)
    for m in exec_list:
        if m.op_type != 'Conv' and m.preserve_quantization:
            m.dtype = m.input_nodes[0].dtype


def pass_propagate_format(graph) -> None:
    """Further propagate output data types.

    Args:
        graph (Graph): The input graph. It will be modified in-place.

    """
    exec_list = sort_graph(graph)
    for m in exec_list:
        if m.op_type != 'Conv' and m.preserve_quantization:
            if m.input_nodes[0].dimension == 'ChHWBCl':
                b = 32
                shape = [(m.channel + b - 1) // b, m.height, m.width, 2, b]
                m.update_shape(shape, m.input_nodes[0].dimension)


def pass_propagate_output_type_backward(graph: Graph) -> None:
    """It is assumed that the output data type of a Graph is float.
       We should propagate this assumption backwards from the output node of the graph to the
       latest quantized convolution available.
    
       There could be cases where the latest convolution node Q is a quantized convolution and we also apply
       thresholds to its outputs. In this cases, the quantized convolution output data type should be float
       even if thresholds are applied.

    Args:
        graph (Graph): The input graph. It will be modified in-place.

    """
    exec_list = sort_graph(graph)

    def output_dtype_changer(node, otype):
        for n in node.input_nodes:
            if n.op_type == 'Conv' and n.is_quantized:
                n.restore_shape()
                n.dtype = otype
                return
            output_dtype_changer(n, otype)

    # propagate output data type to the last quantized convolution
    output_node = exec_list[-1]
    output_type = output_node.dtype
    output_dtype_changer(output_node, output_type)


def pass_lookup(graph: Graph) -> None:
    """Lookup.

    Args:
        graph (Graph): The input graph. It will be modified in-place.

    """
    quantization_types = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'BinaryChannelWiseMeanScalingQuantizer'
    ]

    to_be_removed = []
    exec_list = [n for n in sort_graph(graph) if n.op_type in quantization_types]
    placeholder = [n for n in sort_graph(graph) if n.op_type in 'Input']

    for m in exec_list:
        quantizer = m

        p1 = quantizer.input_nodes[0]
        if p1.op_type != 'Reshape':
            continue
        p2 = p1.input_nodes[0]
        if p2.op_type != 'Reshape':
            continue
        p3 = p2.input_nodes[0]
        if p3.op_type != 'Gather':
            continue
        p4 = p3.input_nodes[0]
        if p4.op_type != 'Gather':
            continue
        gather_params = p4.input_nodes[0]
        if gather_params.rank != 2 or gather_params.shape[0] != 256:
            continue

        params = gather_params.data
        data = {'data': params}
        qtz_data = quantizer.run(**data)['data']

        word_size = 32
        lu_bitwidth = quantizer.nbit
        packer = Packer(lu_bitwidth, word_size)

        lsb = np.zeros((256,), np.uint32)
        msb = np.zeros((256,), np.uint32)

        idx = 0
        for p in qtz_data:
            data = packer.run(p.astype(np.float32), p.shape).flatten()
            lsb[idx] = data[0]
            msb[idx] = data[1]

            idx += 1

        pe_lsb = Constant('pe_lsb_new', QUANTIZED_PACKED_KERNEL(), lsb,
                          dimension_format='TC', packed=True, actual_shape=[256, word_size])
        pe_msb = Constant('pe_msb_new', QUANTIZED_PACKED_KERNEL(), msb,
                          dimension_format='TC', packed=True, actual_shape=[256, word_size])

        n, h, w, c = quantizer.shape
        shape = [1, h, w, 2, word_size]
        pe = Lookup('Lookup', shape, QUANTIZED_PACKED(),
                    {'input': placeholder[0], 'lsb': pe_lsb, 'msb': pe_msb}, dimension_format='ChHWBCl')

        get_nodes_in_branch(quantizer, placeholder[0], to_be_removed)
        placeholder[0].remove_output('output')
        placeholder[0].add_output('output', pe)
        pe.add_outputs(quantizer.output_ops)

        output_op = quantizer.output_op_list[0]

        target_input_name = 'X'
        for input_name in output_op._input_names:
            if quantizer.equals(output_op._input_ops[input_name]):
                target_input_name = input_name
                break

        output_op.add_input(target_input_name, pe)

        graph.add_op(pe_lsb)
        graph.add_op(pe_msb)
        graph.add_op(pe)

    for op in to_be_removed:
        graph.remove_op(op)


def pass_simplify_batchnorm(graph: Graph) -> None:
    """Simplify BarchNorm operator.
    """

    exec_list = [x for x in sort_graph(graph) if x.op_type == 'BatchNormalization']

    to_be_removed = []

    for node in exec_list:
        scale = node.input_ops['scale']
        if scale.op_type != 'Constant':
            raise RuntimeError('scale for BatchNormalization must be Constant')
        B = node.input_ops['B']
        if B.op_type != 'Constant':
            raise RuntimeError('B for BatchNormalization must be Constant')
        mean = node.input_ops['mean']
        if mean.op_type != 'Constant':
            raise RuntimeError('mean for BatchNormalization must be Constant')
        var = node.input_ops['var']
        if var.op_type != 'Constant':
            raise RuntimeError('var for BatchNormalization must be Constant')

        new_name = node.name + '_optimized'
        new_scale_data = scale.data / np.sqrt(var.data + node.epsilon)
        new_scale = Constant(
            new_name + '_scale',
            scale.dtype,
            new_scale_data,
            dimension_format=scale.dimension
        )
        new_bias_data = B.data - new_scale_data * mean.data
        new_bias = Constant(
            new_name + '_bias',
            B.dtype,
            new_bias_data,
            dimension_format=B.dimension
        )
        new_op = BatchNormalizationOptimized(
            new_name,
            node.shape,
            node.dtype,
            {'X': node.input_ops['X'], 'scale': new_scale, 'bias': new_bias},
            dimension_format=node.dimension
        )
        new_scale.add_output('output', new_op)
        new_bias.add_output('output', new_op)

        input_op = node.input_ops['X']
        update_key = None
        new_outputs = [new_op]
        for key, inout_ops in input_op.output_ops.items():
            if node in inout_ops:
                update_key = key
                for op in inout_ops:
                    if op != node:
                        new_outputs.append(op)
        if update_key is not None:
            input_op.remove_output(update_key)
            input_op.add_outputs({update_key: new_outputs})

        out_ops = node.output_op_list
        for op in out_ops:
            update_key = None
            for key, outin_op in op.input_ops.items():
                if outin_op == node:
                    update_key = key
            if update_key is not None:
                op.add_input(update_key, new_op)
            new_op.add_output('Y', op)

        graph.add_op(new_scale)
        graph.add_op(new_bias)
        graph.add_op(new_op)

        to_be_removed += [node, scale, B, mean, var]

    for node in to_be_removed:
        graph.remove_op(node)
