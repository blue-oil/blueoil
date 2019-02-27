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
from lmnet.networks.classification.resnet import ResnetCifarQuantize
from lmnet.datasets.cifar10 import Cifar10
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    Normalize,
)
from lmnet.data_augmentor import (
    Crop,
    FlipLeftRight,
    Pad,
)
from lmnet.quantizations.linear import (
    pact_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = ResnetCifarQuantize
DATASET_CLASS = Cifar10

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 128
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes


MAX_EPOCHS = 200
SAVE_STEPS = 10000
TEST_STEPS = 1000
SUMMARISE_STEPS = 100
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# distributed training
IS_DISTRIBUTION = False

# for debug
# MAX_STEPS = 100
# # BATCH_SIZE = 31
# SAVE_STEPS = 10
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

# https://github.com/facebook/fb.resnet.torch/blob/master/datasets/cifar10.lua
# local meanstd = {
#    mean = {125.3, 123.0, 113.9},
#    std  = {63.0,  62.1,  66.7},
# }

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    Normalize(mean=[125.3, 123.0, 113.9], std=[63.0,  62.1,  66.7])
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
step_per_epoch = int(50000 / BATCH_SIZE)
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.1, 0.01, 0.001],
    "boundaries": [60 * step_per_epoch, 120 * step_per_epoch],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0002
NETWORK.ACTIVATION_QUANTIZER = pact_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 3,
    'initializer': 10.0,
    'decay_rate': NETWORK.WEIGHT_DECAY_RATE,
}

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Pad(4),
    Crop(size=IMAGE_SIZE),
    FlipLeftRight(),
])
