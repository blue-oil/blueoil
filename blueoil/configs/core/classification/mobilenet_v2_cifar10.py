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

from blueoil.common import Tasks
from blueoil.networks.classification.mobilenet_v2 import MobileNetV2
from blueoil.datasets.cifar10 import Cifar10
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from blueoil.data_augmentor import (
    Crop,
    FlipLeftRight,
    Pad,
)


IS_DEBUG = False

NETWORK_CLASS = MobileNetV2
DATASET_CLASS = Cifar10

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 100
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

MAX_EPOCHS = 200
SAVE_CHECKPOINT_STEPS = 100000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 100
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""


# for debug
# MAX_STEPS = 10
# BATCH_SIZE = 31
# SAVE_CHECKPOINT_STEPS = 2
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255(),
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
step_per_epoch = int(50000 / BATCH_SIZE)
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.01, 0.1, 0.01, 0.001, 0.0001],
    "boundaries": [step_per_epoch, step_per_epoch * 100, step_per_epoch * 150, step_per_epoch * 175],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.00004

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
