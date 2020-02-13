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
from datetime import datetime

from tensorflow.io import gfile
import yaml

from executor.train import run as run_train
from blueoil.utils import horovod as horovod_util


def run(config_file, experiment_id):
    """Train from blueoil config.

    Args:
        config_file:
        experiment_id:

    """

    if horovod_util.is_enabled():
        horovod_util.setup()

    # Start training
    run_train(network=None, dataset=None, config_file=config_file, experiment_id=experiment_id, recreate=False)


def train(config, experiment_id=None):
    if not experiment_id:
        # Default model_name will be taken from config file: {model_name}.yml.
        model_name = os.path.splitext(os.path.basename(config))[0]
        experiment_id = '{}_{:%Y%m%d%H%M%S}'.format(model_name, datetime.now())

    run(config, experiment_id)

    output_dir = os.environ.get('OUTPUT_DIR', 'saved')
    experiment_dir = os.path.join(output_dir, experiment_id)
    checkpoint = os.path.join(experiment_dir, 'checkpoints', 'checkpoint')

    if not os.path.isfile(checkpoint):
        raise Exception('Checkpoints are not created in {}'.format(experiment_dir))

    with open(checkpoint) as stream:
        data = yaml.load(stream)
    checkpoint_name = os.path.basename(data['model_checkpoint_path'])

    return experiment_id, checkpoint_name
