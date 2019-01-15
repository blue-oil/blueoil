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
from datetime import datetime
import os

import click

from blueoil.blueoil_init import ask_questions, save_config
from blueoil.blueoil_train import run as run_train
from blueoil.blueoil_convert import run as run_convert, get_export_directory
from executor.predict import run as run_predict


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

    save_config(blueoil_config, output)

    click.echo('')
    click.echo('A new configuration file generated: %s' % (output))
    click.echo('  - Your next step is training.')
    click.echo('  - You can customize some miscellaneous settings according to the comment.')
    click.echo('Config file is generated in {}'.format(output))
    click.echo('Next step: blueoil train -c {}'.format(output))


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
    required=False,
)
def train(config, experiment_id=None):
    if not experiment_id:
        experiment_id = 'train_{:%Y%m%d%H%M%S}'.format(datetime.now())

    run_train(config, experiment_id)

    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    experiment_dir = '{}/{}'.format(output_dir, experiment_id)
    checkpoint_dir = '{}/checkpoints/checkpoint'.format(experiment_dir)
    if not os.path.isfile(checkpoint_dir):
        click.echo('Checkpoints are not created in {}'.format(experiment_dir), err=True)
        exit(1)

    click.echo('Checkpoints are created in {}'.format(experiment_dir))
    click.echo('Next step: blueoil convert -i {} -r {}'.format(experiment_id, experiment_dir))


@main.command(help='Convert trained model to binary files.')
@click.option(
    '-e',
    '--experiment_id',
    help='ID of this experiment.',
    required=True,
)
@click.option(
    '-r',
    '--restore_path',
    help='Restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001',
    required=True,
)
@click.option(
    '-t',
    '--template',
    help='Path of output template directory.',
    envvar='OUTPUT_TEMPLATE_DIR',
    default=None,
)
def convert(experiment_id, restore_path, template):
    run_convert(experiment_id, restore_path, template)

    export_dir = get_export_directory(experiment_id, restore_path)
    output_root_dir = os.path.join(export_dir, 'output')
    click.echo('Output files are generated in {}'.format(output_root_dir))
    click.echo('Please see {}/README.md to run prediction'.format(output_root_dir))


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
    '-r',
    '--restore_path',
    help='Restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001',
    required=True,
)
@click.option(
    "--save_images/--no_save_images",
    help="Flag of saving images. Default is True.",
    default=True,
)
def predict(input, output, experiment_id, restore_path, save_images):
    run_predict(input, output, experiment_id, None, restore_path, save_images)
    click.echo('Result files are created: {}'.format(output))


if __name__ == '__main__':
    main()
