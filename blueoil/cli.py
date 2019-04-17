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
import yaml

from blueoil.blueoil_init import ask_questions, save_config
from blueoil.blueoil_train import run as run_train
from blueoil.blueoil_convert import run as run_convert
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
    required=False,
)
def train(config, experiment_id=None):
    if not experiment_id:
        experiment_id = 'train_{:%Y%m%d%H%M%S}'.format(datetime.now())

    run_train(config, experiment_id)

    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    experiment_dir = os.path.join(output_dir, experiment_id)
    checkpoint = os.path.join(experiment_dir, 'checkpoints', 'checkpoint')
    if not os.path.isfile(checkpoint):
        click.echo('Checkpoints are not created in {}'.format(experiment_dir), err=True)
        exit(1)

    with open(checkpoint) as stream:
        data = yaml.load(stream)
    checkpoint_name = os.path.basename(data['model_checkpoint_path'])

    click.echo('Checkpoints are created in {}'.format(experiment_dir))
    click.echo('Next step: blueoil convert -e {} -p {}'.format(experiment_id, checkpoint_name))


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
    required=True,
)
@click.option(
    '-t',
    '--template',
    help='Path of output template directory.',
    envvar='OUTPUT_TEMPLATE_DIR',
    default=None,
)
def convert(experiment_id, checkpoint, template):
    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    restore_path = os.path.join(output_dir, experiment_id, 'checkpoints', checkpoint)

    export_output_root_dir = run_convert(experiment_id, restore_path, template)

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
    required=True,
)
@click.option(
    "--save_images/--no_save_images",
    help="Flag of saving images. Default is True.",
    default=True,
)
def predict(input, output, experiment_id, checkpoint, save_images):
    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    restore_path = os.path.join(output_dir, experiment_id, 'checkpoints', checkpoint)

    run_predict(input, output, experiment_id, None, restore_path, save_images)

    click.echo('Result files are created: {}'.format(output))


if __name__ == '__main__':
    main()
