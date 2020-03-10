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

- Import protocol buffer, lmnet's export and config.
- Generate all cpp source headers and other control files like Makefile.
"""
import click
from os import path

from blueoil.converter.core.config import Config
from blueoil.converter.core.graph import Graph
from blueoil.converter.core.params import Params
from blueoil.converter.code_generator import CodeGenerator
from blueoil.converter.frontend import TensorFlowIO
from blueoil.converter.core.optimizer import pass_remove_identities, \
    pass_transpose, pass_constant_folding, \
    pass_propagate_quantization_details_into_conv, pass_compute_thresholds, pass_pack_weights, \
    pass_quantize_convolutions, pass_propagate_datatypes, \
    pass_propagate_format, pass_propagate_output_type_backward, \
    pass_lookup, pass_simplify_batchnorm
from blueoil.converter import util


SCRITPS_DIR = path.abspath(path.dirname(__file__))
DLK_ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '..'))
ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '../../..'))


def optimize_graph_step(graph: Graph, config: Config) -> None:
    """Optimizing graph that imported from tensorflow pb.

    Args:
        graph (Graph): Graph that optimization passes are applying to
        config (Config): Collection of configurations

    Returns:


    """

    pass_remove_identities(graph)
    pass_transpose(graph)

    if config.activate_hard_quantization:
        pass_lookup(graph)
        pass_propagate_quantization_details_into_conv(graph)
        if config.threshold_skipping:
            pass_compute_thresholds(graph)
        pass_pack_weights(graph)
        pass_quantize_convolutions(graph)

    if config.threshold_skipping:
        pass_propagate_output_type_backward(graph)
    pass_propagate_datatypes(graph)
    pass_propagate_format(graph)

    pass_constant_folding(graph)
    pass_simplify_batchnorm(graph)

def generate_code_step(graph: Graph, config: Config) -> None:
    """Generate code for the model.

    Args:
        graph (Graph): Graph the code generation is based on
        config (Config): Collection of configurations

    """
    params = Params(graph, config)

    builder = CodeGenerator(graph,
                            params,
                            config)

    builder.reuse_output_buffers()
    builder.generate_files_from_template()
    builder.generate_inputs()

    if config.activate_hard_quantization:
        builder.generate_scaling_factors()

    if config.threshold_skipping:
        builder.generate_thresholds()


def run(input_path: str,
        dest_dir_path: str,
        project_name: str,
        activate_hard_quantization: bool,
        threshold_skipping: bool = False,
        debug: bool = False,
        cache_dma: bool = False):

    output_dlk_test_dir = path.join(dest_dir_path, f'{project_name}.test')
    optimized_pb_path = path.join(dest_dir_path, f'{project_name}')
    optimized_pb_path += '.pb'
    output_project_path = path.join(dest_dir_path, f'{project_name}.prj')

    config = Config(activate_hard_quantization=activate_hard_quantization,
                    threshold_skipping=threshold_skipping,
                    test_dir=output_dlk_test_dir,
                    optimized_pb_path=optimized_pb_path,
                    output_pj_path=output_project_path,
                    debug=debug,
                    cache_dma=cache_dma
    )

    dest_dir_path = path.abspath(dest_dir_path)
    util.make_dirs(dest_dir_path)

    click.echo('import pb file')
    io = TensorFlowIO()
    graph: Graph = io.read(input_path)

    click.echo('optimize graph step: start')
    optimize_graph_step(graph, config)
    click.echo('optimize graph step: done!')

    click.echo('generate code step: start')
    generate_code_step(graph, config)
    click.echo(f'generate code step: done!')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True),
    help="protobuf path which you want to convert to C codes",
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
         debug,
         cache_dma):

    click.echo('start running')
    run(input_path=input_path,
        dest_dir_path=output_path,
        project_name=project_name,
        activate_hard_quantization=activate_hard_quantization,
        threshold_skipping=threshold_skipping,
        debug=debug,
        cache_dma=cache_dma)


if __name__ == '__main__':
    main()
