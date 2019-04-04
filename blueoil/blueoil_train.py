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
import shutil

import click

from blueoil.generate_lmnet_config import generate
from executor.train import run as run_train


def run(blueoil_config_file, experiment_id):
    """Train from blueoil config."""

    # Copy bueoil config yaml.
    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    experiment_dir = os.path.join(output_dir, experiment_id)
    save_config_file(blueoil_config_file, experiment_dir)

    # Generete lmnet config from blueoil config.
    lmnet_config_file = generate(blueoil_config_file)

    # Start training
    run_train(network=None, dataset=None, config_file=lmnet_config_file, experiment_id=experiment_id, recreate=False)


def save_config_file(config_file, dest_dir):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    return shutil.copyfile(
        config_file,
        os.path.join(dest_dir, 'blueoil_config.yaml')
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-c",
    "--config_file",
    help="config file path for this training",
    required=True,
)
@click.option(
    "-i",
    "--experiment_id",
    help="id of this training",
    default="experiment",
    required=True,
)
def main(config_file, experiment_id):
    run(config_file, experiment_id)


if __name__ == '__main__':
    main()
