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
from lmnet.networks.classification.resnet import Resnet
from lmnet.datasets.ilsvrc_2012 import Ilsvrc2012
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    PerImageStandardization,
)
from lmnet.data_augmentor import (
    Crop,
    FlipLeftRight,
)

IS_DEBUG = False

NETWORK_CLASS = Resnet
DATASET_CLASS = Ilsvrc2012

IMAGE_SIZE = [224, 224]
NUM_GPU = 8
BATCH_SIZE = int(256 / NUM_GPU)
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes


# MAX_STEPS = 600000
MAX_EPOCHS = 90
SAVE_STEPS = 50000
TEST_STEPS = 50000
SUMMARISE_STEPS = 10000

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# distributed training
IS_DISTRIBUTION = True

# for debug
# MAX_STEPS = 100
# # BATCH_SIZE = 31
# SAVE_STEPS = 10
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
init_learning_rate = 0.1 * BATCH_SIZE * NUM_GPU / 256
step_per_epoch = int(1280000 / BATCH_SIZE)
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [init_learning_rate, init_learning_rate * 0.1, init_learning_rate * 0.01, init_learning_rate * 0.001],
    "boundaries": [step_per_epoch * 30, step_per_epoch * 60, step_per_epoch * 80],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0001

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Crop(size=IMAGE_SIZE, resize=256),
    FlipLeftRight(),
])
DATASET.ENABLE_PREFETCH = True
