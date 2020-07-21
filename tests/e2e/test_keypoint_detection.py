# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
import pytest

from blueoil.generate_lmnet_config import generate
from conftest import run_all_steps


mscoco2017_single_person_pose_estimation = {
    'model_name': 'mscoco2017_single_person_pose_estimation',
    'task_type': 'keypoint_detection',
    'network_name': 'LmSinglePoseV1Quantize',
    'network': {
        'quantize_first_convolution': False
    },
    'dataset': {
        'format': 'Mscoco for Single-Person Pose Estimation',
        'train_path': 'MSCOCO_2017',
        'test_path': None
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Adam',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [160, 160],
        'pretrain_model': False
    }
}


mscoco2017_single_person_pose_estimation_has_validation = {
    'model_name': 'mscoco2017_single_person_pose_estimation_has_validation',
    'task_type': 'keypoint_detection',
    'network_name': 'LmSinglePoseV1Quantize',
    'network': {
        'quantize_first_convolution': False
    },
    'dataset': {
        'format': 'Mscoco for Single-Person Pose Estimation',
        'train_path': 'MSCOCO_2017',
        'test_path': 'MSCOCO_2017'
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Adam',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [160, 160],
        'pretrain_model': False
    }
}


@pytest.mark.parametrize(
    "config", [
        mscoco2017_single_person_pose_estimation,
        mscoco2017_single_person_pose_estimation_has_validation,
    ]
)
def test_keypoint_detection(init_env, config):
    """Run Blueoil test of keypoint detection"""
    config_file = generate(config)
    run_all_steps(init_env, config_file)
