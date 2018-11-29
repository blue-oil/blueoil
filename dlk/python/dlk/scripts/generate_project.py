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

from core.config import Config
from core.graph import Graph
from core.model import Model
from core.params import Params
from core.optimizer import Optimizer
from code_generater import CodeGenerater
from frontend import TensorFlowIO
from core.graph_pattern_matching import GraphMatcher, Pattern
from core.operators import Constant

import utils

SCRITPS_DIR = path.abspath(path.dirname(__file__))
DLK_ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '..'))
ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '../../..'))


def pass_print(graph: Graph, name=str()):

    gm = GraphMatcher(graph)

    print('--- ', name, '---')
    matches = list()
    p = Pattern("*")
    gm.get_op_type_matches(p, matches)

    for m in matches:
        print('Match: ', m.node.name, m.node.op_type, m.node.dimension)
        for input_node in m.node.input_nodes:
            print('   -> ', input_node.name, input_node.op_type)

    print('---')


def pass_dot_graph(graph: Graph, filename):

    dot_script = 'digraph {'

    code = {}
    counter = 0
    for node in graph.operators:
        code[node.name] = counter
        counter += 1

    for node in graph.operators:
        for input_node in node.input_nodes:
            quant = node.quantizer.name if node.op_type == 'Conv' and node.quantizer else 'None'
            aquant = node.a_quantizer[0].name if node.op_type == 'Conv' and node.a_quantizer else 'None'

            dot_script += '"' + format(code[input_node.name], '04X') + '-' + input_node.op_type + '"' + ' -> ' \
                        + '"' + format(code[node.name], '04X') + '-' + node.op_type + '-' + aquant + '/' + quant + '"' + ';'

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
        # print('Match: ', m.node.name, m.node.op_type)
        # for input_node in m.node.input_nodes:
        #     print('   -> ', input_node.name, input_node.op_type)

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
        # print('Match: ', m.node.name, m.node.op_type)
        # for input_node in m.node.input_nodes:
        #     print('   -> ', input_node.name, input_node.op_type)

        dim = m.node.dimension
        shape = m.node.shape
        if len(shape) != 4 or len(dim) != 4 or not set(dim).issubset({'N', 'H', 'W', 'C', 'I', 'O'}):
            continue

        dim = dim.replace('I', 'C')
        dim = dim.replace('O', 'N')

        permutation = list(map(lambda s: dim.index(s), 'NHWC'))
        m.node.transpose(permutation)


def pass_precompute(graph) -> int:

    gm = GraphMatcher(graph)

    to_be_removed = list()
    matches = list()
    p = Pattern("*")
    gm.get_op_type_matches(p, matches)

    for m in matches:

        # We want operators with inputs
        if not m.node.input_nodes:
            continue

        # Leave out nodes which execution will lose information.
        # They will have a special processing later.
        if m.node.run_it_will_lose_information:
            continue

        precomputable = True
        for input_node in m.node.input_nodes:
            if input_node.op_type != 'Constant':
                precomputable = False

        if not precomputable:
            continue

        to_be_removed += m.node.input_nodes
        to_be_removed.append(m.node)

        m.node.run_forward()

        new_constant = Constant(
            m.node.name + '_new',
            m.node.dtype,
            m.node.data,
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

    for op in to_be_removed:
        graph.remove_op(op)

    return len(to_be_removed)


def pass_propagate_quantization_details_into_conv(graph):

    gm = GraphMatcher(graph)

    matches = list()
    p = Pattern('*')
    gm.get_op_type_matches(p, matches)

    quantization_types = [
        'QTZ_binary_mean_scaling',
        'QTZ_linear_mid_tread_half',
        'QTZ_binary_channel_wise_mean_scaling'
    ]

    quantization_details = {}
    for m in matches:
        if not m.node.preserve_quantization:
            quantization_details[m.node.name] = None
            continue

        current_node_quant_details = []
        for input_node in m.node.input_nodes:
            if input_node.op_type in quantization_types:
                current_node_quant_details.append(input_node)
            else:
                current_node_quant_details.append(quantization_details[input_node.name])

        if m.node.op_type == 'Conv':
            m.node.a_quantizer = [current_node_quant_details[0]] if current_node_quant_details[0] else []
            m.node.quantizer = current_node_quant_details[1]
            quantization_details[m.node.name] = None
        else:
            all_quantizers = True
            for quantizer in current_node_quant_details:
                if not quantizer:
                    all_quantizers = False
                    break

            if not all_quantizers:
                same_nbits = False
            else:
                same_nbits = all(quantizer.nbit == current_node_quant_details[0].nbit
                                 for quantizer in current_node_quant_details)

            quantization_details[m.node.name] = current_node_quant_details[0] if same_nbits else None

            if not same_nbits:
                print(f'Warning: Not every input node of {m.node.name} is quantized to the same bit-width')


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

    pass_print(graph, 'Before')
    pass_dot_graph(graph, '/tmp/original.dot')

    pass_remove_identities(graph)
    pass_print(graph, 'After identity')
    pass_dot_graph(graph, '/tmp/prune_identities.dot')

    pass_transpose(graph)
    pass_print(graph, 'After transpose')
    pass_dot_graph(graph, '/tmp/transposed.dot')

    pass_precompute(graph)
    pass_print(graph, 'After precompute')

    pass_propagate_quantization_details_into_conv(graph)
    pass_print(graph, 'After propagate')

    pass_dot_graph(graph, '/tmp/final.dot')

    optim = Optimizer()
    optim.transpose_NHWC(graph)
    optim.precompute(graph, config.activate_hard_quantization)
    if config.threshold_skipping:
        optim.threshold_skipping(graph)


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
