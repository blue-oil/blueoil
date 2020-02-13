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
import os.path

TMP_DIR = "tmp"
LOG_DIR = os.path.join(TMP_DIR, "log")

default_data_dir = "dataset"
# DATA_DIR = os.getenv("DATA_DIR", default_data_dir)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), default_data_dir))

default_output_dir = "saved"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", default_output_dir)

_EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}")
_TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}", "tensorboard")
_CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}", "checkpoints")

_init_flag = False


def init(experiment_id):
    """Initialize experiment environment.

    experiment id embed to directories.
    """
    global _init_flag, EXPERIMENT_DIR, TENSORBOARD_DIR, CHECKPOINTS_DIR

    if _init_flag:
        raise Exception("Experiment setting already initialized.")

    else:
        # remove OUTPUT_DIR if it be included in experiment_id.
        if OUTPUT_DIR + os.path.sep in experiment_id:
            experiment_id = experiment_id.replace(OUTPUT_DIR + os.path.sep, "")
            print(experiment_id)

        EXPERIMENT_DIR = _EXPERIMENT_DIR.format(experiment_id=experiment_id)

        # directory to save this experiment outputs for tensorboard.
        TENSORBOARD_DIR = _TENSORBOARD_DIR.format(experiment_id=experiment_id)

        # checkpoints_dir in the same way of tensorboard_dir.
        CHECKPOINTS_DIR = _CHECKPOINTS_DIR.format(experiment_id=experiment_id)

        _init_flag = True


def setup_test_environment():
    """Override `OUTPUT_DIR` and `DATA_DIR` for test."""
    global _init_flag, DATA_DIR, _EXPERIMENT_DIR, _TENSORBOARD_DIR, _CHECKPOINTS_DIR

    _init_flag = False

    DATA_DIR = "tests/fixtures/datasets"

    OUTPUT_DIR = "tmp/tests/saved"

    _EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}")
    _TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}", "tensorboard")
    _CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}", "checkpoints")


def teardown_test_environment():
    """Reset test environment."""
    global _init_flag, DATA_DIR, _EXPERIMENT_DIR, _TENSORBOARD_DIR, _CHECKPOINTS_DIR

    _init_flag = False

    default_data_dir = "dataset"
    DATA_DIR = os.getenv("DATA_DIR", default_data_dir)

    default_output_dir = "saved"
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", default_output_dir)

    _EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}")
    _TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}", "tensorboard")
    _CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "{experiment_id}", "checkpoints")
