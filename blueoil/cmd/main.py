#!/usr/bin/env python
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
import os

import click

from blueoil.cmd.convert import convert as run_convert
from blueoil.cmd.init import ask_questions, save_config
from blueoil.cmd.predict import predict as run_predict
from blueoil.cmd.train import train as run_train


@click.group(invoke_without_command=True)
@click.pass_context
def main(context):
    if context.invoked_subcommand is None:
        click.echo(context.get_help())


@main.command(help='Generate blueoil config.')
@click.option(
    '-o',
    '--output',
    help='Path of generated configuration file.',
    required=False,
)
def init(output):
    blueoil_config = ask_questions()

    if output:
        output_dir = os.path.dirname(os.path.abspath(output))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    config_filepath = save_config(blueoil_config, output)

    click.echo('')
    click.echo('A new configuration file generated: %s' % (config_filepath))
    click.echo('  - Your next step is training.')
    click.echo('  - You can customize some miscellaneous settings according to the comment.')
    click.echo('Config file is generated in {}'.format(config_filepath))
    click.echo('Next step: blueoil train -c {}'.format(config_filepath))


@main.command(help='Run training.')
@click.option(
    '-c',
    '--config',
    help='Path of config file.',
    required=True,
)
@click.option(
    '-e',
    '--experiment_id',
    help='ID of this training.',
    default=None,
)
@click.option(
    '--recreate',
    is_flag=True,
    help='Delete and recreate experiment id dir',
    default=False,
)
@click.option(
    '-n',
    '--network',
    help='Network name which you want to use for this training. override config.DATASET_CLASS',
    default=None,
)
@click.option(
    '-d',
    '--dataset',
    help='Dataset name which is the source of this training. override config.NETWORK_CLASS',
    default=None,
)
def train(config, experiment_id, recreate, network, dataset):
    experiment_id, checkpoint_name = run_train(config, experiment_id, recreate, network, dataset)
    click.echo('Next step: blueoil convert -e {} -p {}'.format(
        experiment_id,
        checkpoint_name
    ))

@main.command(help='Convert trained model to binary files.')
@click.option(
    '-e',
    '--experiment_id',
    help='ID of this experiment.',
    required=True,
)
@click.option(
    '-p',
    '--checkpoint',
    help='Checkpoint name. e.g. save.ckpt-10001',
    default=None,
)
@click.option(
    '-t',
    '--template',
    help='Path of output template directory.',
    envvar='OUTPUT_TEMPLATE_DIR',
    default=None,
)
@click.option(
    "--image_size",
    nargs=2,
    type=click.Tuple([int, int]),
    help="input image size height and width. if these are not provided, it restores from saved experiment config."
    "e.g --image_size 320 320",
    default=(None, None),
)
@click.option(
    "--project_name",
    help="project name which generated by convert",
    default=None,
)
def convert(experiment_id, checkpoint, template, image_size, project_name):
    export_output_root_dir = run_convert(experiment_id, checkpoint, template, image_size, project_name)

    click.echo('Output files are generated in {}'.format(export_output_root_dir))
    click.echo('Please see {}/README.md to run prediction'.format(export_output_root_dir))


@main.command(help='Predict by using trained model.')
@click.option(
    '-i',
    '--input',
    help='Directory containing predicted images.',
    required=True,
)
@click.option(
    '-o',
    '--output',
    help='Directory to output prediction result.',
    required=True,
)
@click.option(
    '-e',
    '--experiment_id',
    help='ID of this experiment.',
    required=True,
)
@click.option(
    '-p',
    '--checkpoint',
    help='Checkpoint name. e.g. save.ckpt-10001',
    default=None,
)
@click.option(
    "--save_images/--no_save_images",
    help="Flag of saving images. Default is True.",
    default=True,
)
def predict(input, output, experiment_id, checkpoint, save_images):
    run_predict(input, output, experiment_id, checkpoint, save_images)

    click.echo('Result files are created: {}'.format(output))


if __name__ == '__main__':
    main()
