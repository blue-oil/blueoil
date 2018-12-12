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
"""
Script that automatically runs all of the folllowing steps.

- Import onnx, lmnet's export and config.
- Generate all cpp source headers and other control files like Makefile.
"""
import click
from os import path
import shutil
import math
import numpy as np

from core.config import Config
from core.graph import Graph
from core.model import Model
from core.params import Params
from core.optimizer import Optimizer
from code_generater import CodeGenerater
from frontend import TensorFlowIO
from core.graph_pattern_matching import GraphMatcher, Pattern, match_to_execution_list
from core.operators import Constant, Operator
from modules.packer import Packer
from core.data_types import Uint32, QUANTIZED_NOT_PACKED
from typing import cast
from collections import defaultdict
import utils

SCRITPS_DIR = path.abspath(path.dirname(__file__))
DLK_ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '..'))
ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '../../..'))


def pass_dot_graph(graph: Graph, filename):

    dot_script = 'digraph {'

    code = {}
    counter = 0
    for node in graph.operators:
        code[node.name] = counter
        counter += 1

    for node in graph.operators:

        shape = '-'
        if node.shape:
            shape = 'x'.join(str(x) for x in node.shape)
        shape += '(' + node.dimension + ')'

        dot_script += node.name + '[label="<f0> ' + format(code[node.name], '04X') + '| <f1> ' + \
            node.op_type + '| <f2> ' + shape + '| <f3> ' + node.dtype.cpptype() + '" shape = "record"];'
        for i in node.input_nodes:
            dot_script += i.name + ' -> ' + node.name + ';'

    dot_script += '}'

    with open(filename, 'w') as f:
        f.write(dot_script)


def pass_remove_identities(graph: Graph):

    gm = GraphMatcher(graph)

    to_be_removed = list()
    matches = list()
    p = Pattern("Identity")
    gm.get_op_type_matches(p, matches)

    for m in matches:
        """skip all identity."""
        in_op = m.node.input_ops['input']
        out_ops = m.node.output_ops['output']
        for out_op in out_ops:
            for k, v in out_op.input_ops.items():
                if v == m.node:
                    # change the output's input to this identity's input
                    out_op.add_input(k, in_op)
                    # change the input's output to this identity's output
                    for k2, v2 in in_op.output_ops.items():
                        if m.node in v2:
                            v2.remove(m.node)
                            v2.append(out_op)
                            break
                    break

        to_be_removed.append(m.node)

    for op in to_be_removed:
        graph.remove_op(op)


def pass_transpose(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern("*")
    gm.get_op_type_matches(p, matches)

    for m in matches:
        dim = m.node.dimension
        shape = m.node.shape
        if len(shape) != 4 or len(dim) != 4 or not set(dim).issubset({'N', 'H', 'W', 'C', 'I', 'O'}):
            continue

        dim = dim.replace('I', 'C')
        dim = dim.replace('O', 'N')

        permutation = list(map(lambda s: dim.index(s), 'NHWC'))
        m.node.transpose(permutation)


def pass_precompute(graph, processed_nodes):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('*')
    gm.get_op_type_matches(p, matches)

    processed_before_precompute = len(processed_nodes)

    for m in matches:
        if m.node in processed_nodes:
            continue

        # We want operators with inputs
        if not m.node.input_nodes:
            continue

        precomputable = True
        for input_node in m.node.input_nodes:
            if input_node.op_type != 'Constant':
                precomputable = False

        if not precomputable:
            continue

        processed_nodes += m.node.input_nodes
        processed_nodes.append(m.node)

        data = m.node.run_forward()

        new_constant = Constant(
            m.node.name + '_new',
            m.node.dtype,
            data,
            dimension_format=m.node.dimension
        )

        graph.add_op(new_constant)

        new_constant.add_outputs(m.node.output_ops)
        for output_name, consumer_list in m.node.output_ops.items():
            for consumer_node in consumer_list:
                for input_name, input_node in consumer_node.input_ops.items():
                    if input_node == m.node:
                        consumer_node.add_input(input_name, new_constant)
                        break

    return len(processed_nodes) > processed_before_precompute


def pass_propagate_quantization_details_into_conv(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('*')
    gm.get_op_type_matches(p, matches)

    qtypes = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'QTZ_binary_channel_wise_mean_scaling'
    ]

    quant_details = defaultdict(list)
    for m in matches:
        if not m.node.preserve_quantization:
            quant_details[m.node.name] = []
            continue

        if m.node.op_type == 'Conv':
            input_node = m.node.input_nodes[0]
            weight_node = m.node.input_nodes[1]

            m.node.a_quantizer = [input_node] if input_node.op_type in qtypes else quant_details[input_node.name]
            m.node.quantizer = weight_node if weight_node.op_type in qtypes else quant_details[weight_node.name]

            quant_details[m.node.name] = []
        else:
            qtzs = []
            for n in m.node.input_nodes:
                if n.op_type in qtypes:
                    qtzs.append(n)
                else:
                    for q in quant_details[n.name]:
                        qtzs.append(q)

            quant_details[m.node.name] = qtzs if len(qtzs) == len(m.node.input_nodes) else []
            # TODO: check if the quantizers use same n_bits


def pass_compute_thresholds(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('QTZ_linear_mid_tread_half')

    gm.get_op_type_matches(p, matches)

    for m in matches:

        p = [m.node]
        while p[-1].op_type != 'Conv':
            non_variable_input = [inode for inode in p[-1].input_nodes
                                  if (not cast(Operator, inode).is_variable and inode.is_monotonic)
                                  or inode.op_type == 'Conv']
            if len(non_variable_input) != 1:
                break
            p.append(non_variable_input[-1])

        if p[-1].op_type != 'Conv':
            continue
        quantizer_conv_output_node = p[0]
        conv_node = p[-1]

        # check if this is a quantized convolution
        if not conv_node.quantizer or not conv_node.a_quantizer:
            continue

        quantizer_conv_weights = conv_node.quantizer
        quantizer_conv_weights.run_forward_no_scaling_factor()
        scaling_factor = quantizer_conv_weights.scaling_factor

        # TODO: make '3' function on the number of bits of the number of bits
        n = 2 ** 2 - 1
        ch = conv_node.channel
        # assume that the threshold values will be a 13-bit signed integer
        max_th_value = 2 ** 12 - 1

        # The threshold_table is ndarray that holds the threshold values for all channels
        threshold_table = np.empty([ch, n + 1], dtype=np.int32)

        # Compute threshold (t0, t1, t2)
        for th_id, th_v in enumerate([0.5, 1.5, 2.5]):
            init_threshold = np.full(ch, th_v, dtype=np.float64)

            # run calculation in reverse order: q -> bn -> scaling
            trans_th = {'data': init_threshold}
            for op in p[:-1]:
                trans_th = op.de_run(**trans_th)
            threshold = (trans_th['data'] * np.float64(n)) / (np.float64(2.0) * scaling_factor)

            for ch_id, th_per_ch in enumerate(threshold):
                if quantizer_conv_weights.op_type == 'QTZ_binary_channel_wise_mean_scaling':
                    threshold_table[ch_id, th_id] = int(math.floor(th_per_ch)) \
                        if (scaling_factor[ch_id] < 0) ^ (ch_id in trans_th['nega_idx']) \
                        else int(math.ceil(th_per_ch))
                else:
                    threshold_table[ch_id, th_id] = int(math.floor(th_per_ch)) \
                        if (scaling_factor < 0) ^ (ch_id in trans_th['nega_idx']) \
                        else int(math.ceil(th_per_ch))

        # take care of threshold values that are larger than 16-bit signed integer
        threshold_table[abs(threshold_table) > max_th_value] = max_th_value

        for c in range(ch):
            threshold_table[c, -1] = 1 \
                if np.all(threshold_table[c, 1:-1] > threshold_table[c, :-2], axis=0) else -1
            # Applying the magic number
            if np.all(threshold_table[c, 1:-1] == threshold_table[c, :-2], axis=0):
                threshold_table[c, -1] = 2

        # Put the thresholds into list
        conv_node.thresholds = threshold_table.flatten().tolist()

        # TODO: Neil-san, you should keep this
        # Disconnect batchnorm and the quantizer
        out_ops = quantizer_conv_output_node.output_ops['output']
        for output_node in out_ops:
            for input_name, input_node in output_node.input_ops.items():
                if input_node == quantizer_conv_output_node:
                    output_node.add_input(input_name, conv_node)

        conv_node.remove_output('Y')
        conv_node.add_outputs({'Y': out_ops})

        # TODO: temporary (only for drawing better graphs)
        # batch_norm_node.remove_input('X')


def pass_pack_weights(graph):

    gm = GraphMatcher(graph)

    quantization_types = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'QTZ_binary_channel_wise_mean_scaling'
    ]

    matches = list()
    p = Pattern('Conv')

    gm.get_op_type_matches(p, matches)

    # TODO: pass proper parameters
    packer = Packer(1, 32)

    for m in matches:
        conv_node = m.node

        # check if this is a quantized convolution
        if not conv_node.quantizer or not conv_node.a_quantizer:
            continue

        weight_quantizer = conv_node.quantizer
        if weight_quantizer.op_type not in quantization_types:
            continue

        # Quantize the weights
        weight_quantizer.run_forward()
        op_data = weight_quantizer.binarizer(weight_quantizer.data)
        data = packer.run(op_data.astype(np.float32), weight_quantizer.dimension)

        quantized_constant = Constant(
            weight_quantizer.name + '_new',
            Uint32(),
            data,
            packed=True,
            actual_shape=weight_quantizer.shape
        )

        graph.add_op(quantized_constant)

        quantized_constant.add_outputs(weight_quantizer.output_ops)
        for output_name, consumer_list in weight_quantizer.output_ops.items():
            for consumer_node in consumer_list:
                for input_name, input_node in consumer_node.input_ops.items():
                    if input_node == weight_quantizer:
                        consumer_node.add_input(input_name, quantized_constant)
                        break


def pass_quantize_convolutions(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('Conv')
    gm.get_op_type_matches(p, matches)

    for m in matches:
        conv_node = m.node

        # check if this is a quantized convolution
        if not conv_node.quantizer or not conv_node.a_quantizer:
            continue

        # Mark as quantized convolution
        conv_node.is_quantized = True

        # change the output data type of the convolution if thresholds are available
        if conv_node.has_thresholds:
            conv_node.dtype = QUANTIZED_NOT_PACKED

        # change the output data type of the quantizers
        conv_node.quantizer.dtype = Uint32
        for qtz in conv_node.a_quantizer:
            qtz.dtype = QUANTIZED_NOT_PACKED


def pass_propagate_datatypes(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('*')
    gm.get_op_type_matches(p, matches)

    for m in matches:
        if m.node.op_type != 'Conv' and m.node.preserve_quantization:
            m.node.dtype = m.node.input_nodes[0].dtype


def pass_propagate_output_type_backward(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('*')

    gm.get_op_type_matches(p, matches)

    def find_input(node, otype):
        for n in node.input_nodes:
            if n.op_type == 'Conv' and n.is_quantized:
                n.dtype = otype
                return
            find_input(n, otype)

    output_node = matches[-1].node

    output_type = output_node.dtype
    find_input(output_node, output_type)


def optimize_graph_step(model: Model, config: Config) -> None:
    """Optimze graph in the model.

    Parameters
    ----------
    model : Model
        Model that contains the graph

    config : Config
        Collection of configurations

    """
    graph: Graph = model.graph

    pass_dot_graph(graph, '/tmp/original.dot')

    pass_remove_identities(graph)
    pass_dot_graph(graph, '/tmp/prune_identities.dot')

    pass_transpose(graph)
    pass_dot_graph(graph, '/tmp/transposed.dot')

    if config.activate_hard_quantization:
        pass_propagate_quantization_details_into_conv(graph)
        if config.threshold_skipping:
            pass_compute_thresholds(graph)
            # pass_propagate_output_type_backward(graph)
        pass_pack_weights(graph)
        pass_quantize_convolutions(graph)

    if config.threshold_skipping:
        pass_propagate_output_type_backward(graph)
    pass_propagate_datatypes(graph)

    processed_nodes = []
    while pass_precompute(graph, processed_nodes=processed_nodes):
        pass
    pass_dot_graph(graph, '/tmp/final.dot')

    optim = Optimizer()
    # optim.transpose_NHWC(graph)
    # optim.precompute(graph, config.activate_hard_quantization)
    # if config.threshold_skipping:
    #    optim.threshold_skipping(graph)


def generate_code_step(model: Model, config: Config) -> None:
    """Generate code for the model.

    Parameters
    ----------
    model : Model
        Model the code generation is based on

    config : Config
        Collection of configurations

    """
    graph: Graph = model.graph
    params = Params(graph, config)

    builder = CodeGenerater(graph,
                            params,
                            config)

    builder.reuse_output_buffers()
    builder.generate_files_from_template()
    builder.generate_inputs()

    if config.activate_hard_quantization:
        builder.generate_scaling_factors()

    if config.threshold_skipping:
        builder.generate_thresholds()

    if config.use_tvm:
        builder.generate_tvm_libraries()


def run(input_path: str,
        dest_dir_path: str,
        project_name: str,
        activate_hard_quantization: bool,
        threshold_skipping: bool = False,
        num_pe: int = 16,
        use_tvm: bool = False,
        use_onnx: bool = False,
        debug: bool = False,
        cache_dma: bool = False):

    output_dlk_test_dir = path.join(dest_dir_path, f'{project_name}.test')
    optimized_pb_path = path.join(dest_dir_path, f'{project_name}')
    optimized_pb_path += '.onnx' if use_onnx else '.pb'
    output_project_path = path.join(dest_dir_path, f'{project_name}.prj')

    config = Config(num_pe=num_pe,
                    activate_hard_quantization=activate_hard_quantization,
                    threshold_skipping=threshold_skipping,
                    tvm_path=(path.join(ROOT_DIR, 'tvm') if use_tvm else None),
                    test_dir=output_dlk_test_dir,
                    optimized_pb_path=optimized_pb_path,
                    output_pj_path=output_project_path,
                    debug=debug,
                    cache_dma=cache_dma
                    )

    dest_dir_path = path.abspath(dest_dir_path)
    utils.make_dirs(dest_dir_path)

    click.echo('import pb file')
    if use_onnx:
        try:
            __import__('onnx')
        except ImportError:
            raise ImportError('ONNX is required but not installed.')
        from frontend.base import BaseIO
        from frontend.onnx import OnnxIO
        io: BaseIO = OnnxIO()

    else:
        io = TensorFlowIO()
    model: Model = io.read(input_path)

    click.echo('optimize graph step: start')
    optimize_graph_step(model, config)
    click.echo('optimize graph step: done!')

    click.echo('generate code step: start')
    generate_code_step(model, config)
    click.echo(f'generate code step: done!')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True),
    help="onnx protobuf path which you want to convert to C codes",
)
@click.option(
    "-o",
    "--output_path",
    help="output path which you want to export any generated files",
)
@click.option(
    "-p",
    "--project_name",
    help="project name which you'll generate",
)
@click.option(
    "-hq",
    "--activate_hard_quantization",
    is_flag=True,
    default=False,
    help="activate hard quantization optimization",
)
@click.option(
    "-ts",
    "--threshold_skipping",
    is_flag=True,
    default=False,
    help="activate threshold skip optimization",
)
@click.option(
    '-pe',
    '--num_pe',
    type=int,
    default=16,
    help='set number of PE used in FPGA IP',
)
@click.option(
    "-tvm",
    "--use_tvm",
    is_flag=True,
    default=False,
    help="optimize CPU/GPU operations using TVM",
)
@click.option(
    "-onnx",
    "--use_onnx",
    is_flag=True,
    default=False,
    help="if the input file is in ONNX format"
)
@click.option(
    "-dbg",
    "--debug",
    is_flag=True,
    default=False,
    help="add debug code to the generated project",
)
@click.option(
    "-cache",
    "--cache_dma",
    is_flag=True,
    default=False,
    help="use cached DMA buffers",
)
def main(input_path,
         output_path,
         project_name,
         activate_hard_quantization,
         threshold_skipping,
         num_pe,
         use_tvm,
         use_onnx,
         debug,
         cache_dma):

    click.echo('start running')
    run(input_path=input_path,
        dest_dir_path=output_path,
        project_name=project_name,
        activate_hard_quantization=activate_hard_quantization,
        threshold_skipping=threshold_skipping,
        num_pe=num_pe,
        use_tvm=use_tvm,
        use_onnx=use_onnx,
        debug=debug,
        cache_dma=cache_dma)


if __name__ == '__main__':
    main()
