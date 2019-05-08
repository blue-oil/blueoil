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
from lmnet.networks.segmentation.lm_segnet_v1 import LmSegnetV1Quantize
from lmnet.datasets.camvid import Camvid
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)
from hyperopt import hp

IS_DEBUG = False

NETWORK_CLASS = LmSegnetV1Quantize
DATASET_CLASS = Camvid

IMAGE_SIZE = [360, 480]
BATCH_SIZE = 8
DATA_FORMAT = "NHWC"
TASK = Tasks.SEMANTIC_SEGMENTATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 150000
SAVE_STEPS = 3000
TEST_STEPS = 1000
SUMMARISE_STEPS = 1000

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

TUNE_SPEC = {
        'run': 'tunable',
        'resources_per_trial': {"cpu": 2, "gpu": 1},
        'stop': {
            'mean_accuracy': 1.0,
            'training_iteration': 200,
        },
        'config': {
            'lm_config': {},
        },
        'local_dir': None,
        'num_samples': 100,
}

TUNE_SPACE = {
    'optimizer_class': hp.choice(
        'optimizer_class', [
            {
                'optimizer': tf.train.AdamOptimizer,
            },
        ]
    ),
    'learning_rate': hp.uniform('learning_rate', 0, 0.01),
    'learning_rate_func': hp.choice(
        'learning_rate_func', [
            {
                'scheduler': tf.train.piecewise_constant,
                'scheduler_factor': 1.0,
                'scheduler_steps': [25000, 50000, 75000],
            },
        ]
    ),
}

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = None
NETWORK.OPTIMIZER_KWARGS = {}
NETWORK.LEARNING_RATE_FUNC = None
NETWORK.LEARNING_RATE_KWARGS = {}
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

DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    FlipLeftRight(),
    Hue((-10, 10)),
])

# TODO(Neil): dataset pre-fectch is not supported at the moment
DATASET.ENABLE_PREFETCH = False
