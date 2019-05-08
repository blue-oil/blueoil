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
from easydict import EasyDict
import tensorflow as tf

from lmnet.common import Tasks
from lmnet.networks.classification.lmnet_v1 import LmnetV1Quantize
from lmnet.datasets.cifar10 import Cifar10
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from lmnet.data_augmentor import (
    Crop,
    FlipLeftRight,
    Pad,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

from hyperopt import hp

IS_DEBUG = False

NETWORK_CLASS = LmnetV1Quantize
DATASET_CLASS = Cifar10

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 100
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 100000
SAVE_STEPS = 5000
TEST_STEPS = 1000
SUMMARISE_STEPS = 100

# distributed training
IS_DISTRIBUTION = False

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255(),
])
POST_PROCESSOR = None

STEP_PER_EPOCH = int(50000 / BATCH_SIZE)

TUNE_SPEC = {
        'run': 'tunable',
        'resources_per_trial': {"cpu": 2, "gpu": 0.5},
        'stop': {
            'mean_accuracy': 0.87,
            'training_iteration': 200,
        },
        'config': {
            'lm_config': {},
        },
        "local_dir": None,
        "num_samples": 300,
}

TUNE_SPACE = {
    'optimizer_class': hp.choice(
        'optimizer_class', [
            {
                'optimizer': tf.train.MomentumOptimizer,
                'momentum': 0.9,
            },
        ]
    ),
    'learning_rate': hp.uniform('learning_rate', 0, 0.01),
    'learning_rate_func': hp.choice(
        'learning_rate_func', [
            {
                'scheduler': tf.train.piecewise_constant,
                'scheduler_factor': hp.uniform('scheduler_factor', 0.05, 0.5),
                'scheduler_steps': [25000, 50000, 75000],
            },
        ]
    ),
    'weight_decay_rate': 0.0001,
}

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = None
NETWORK.OPTIMIZER_KWARGS = {}
NETWORK.LEARNING_RATE_FUNC = None
NETWORK.LEARNING_RATE_KWARGS = {}
NETWORK.WEIGHT_DECAY_RATE = None
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Pad(2),
    Crop(size=IMAGE_SIZE),
    FlipLeftRight(),
])
DATASET.TRAIN_VALIDATION_SAVING_SIZE = 5000
