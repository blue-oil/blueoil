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
from easydict import EasyDict
import tensorflow as tf

from blueoil.common import Tasks
from blueoil.networks.classification.lm_resnet import LmResnetQuantize
from blueoil.datasets.cifar10 import Cifar10
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    PerImageStandardization,
)
from blueoil.data_augmentor import (
    Crop,
    FlipLeftRight,
    Pad,
)

from blueoil.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LmResnetQuantize
DATASET_CLASS = Cifar10

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 128
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes


MAX_STEPS = 100000
SAVE_CHECKPOINT_STEPS = 2500
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 2500
SUMMARISE_STEPS = 2500
USE_RECOVERY = True
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""


# for debug
# MAX_STEPS = 100
# # BATCH_SIZE = 31
# SAVE_CHECKPOINT_STEPS = 10
# KEEP_CHECKPOINT_MAX = 5
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.1, 0.01, 0.001, 0.0001],
    "boundaries": [40000, 60000, 80000],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0001
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
