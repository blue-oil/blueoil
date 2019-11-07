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

from blueoil.generate_lmnet_config import generate
from executor.train import run as run_train
from lmnet.utils import horovod as horovod_util


def run(blueoil_config_file, experiment_id):
    """Train from blueoil config.

    Args:
        blueoil_config_file: 
        experiment_id: 

    """

    if horovod_util.is_enabled():
        horovod_util.setup()

    if horovod_util.is_rank0():
        # Copy bueoil config yaml.
        output_dir = os.environ.get('OUTPUT_DIR', 'saved')
        experiment_dir = os.path.join(output_dir, experiment_id)
        save_config_file(blueoil_config_file, experiment_dir)

    # Generete lmnet config from blueoil config.
    # this lmnet_config_file cannot be reuse from multiprocesses as the file is a named temporary file.
    lmnet_config_file = generate(blueoil_config_file)

    # Start training
    run_train(network=None, dataset=None, config_file=lmnet_config_file, experiment_id=experiment_id, recreate=False)


def save_config_file(config_file, dest_dir):
    if not gfile.exists(dest_dir):
        gfile.makedirs(dest_dir)

    config_file_dest = os.path.join(dest_dir, 'blueoil_config.yaml')

    # HACK: This is for tensorflow bug workaround.
    # We can remove following 2 lines once it's been resolved in tensorflow
    # issue link: https://github.com/tensorflow/tensorflow/issues/28508
    if gfile.exists(config_file_dest):
        gfile.Remove(config_file_dest)

    return gfile.Copy(
        config_file,
        config_file_dest
    )


def train(config, experiment_id=None):
    if not experiment_id:
        experiment_id = 'train_{:%Y%m%d%H%M%S}'.format(datetime.now())

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
