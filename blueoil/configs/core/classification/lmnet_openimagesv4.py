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
from blueoil.networks.classification import Lmnet
from blueoil.datasets.open_images_v4 import OpenImagesV4Classification
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    PerImageStandardization,
)
from blueoil.data_augmentor import (
    Crop,
    FlipLeftRight,
)

IS_DEBUG = False

NETWORK_CLASS = Lmnet
DATASET_CLASS = OpenImagesV4Classification
CLASSES = DATASET_CLASS(subset="train", batch_size=1).classes

IMAGE_SIZE = [32, 32]
BATCH_SIZE = 200
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION

MAX_STEPS = 100000
SAVE_CHECKPOINT_STEPS = 100000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 1000
SUMMARISE_STEPS = 100
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization()
])
POST_PROCESSOR = None


# for debug
# MAX_STEPS = 10
# BATCH_SIZE = 31
# SAVE_CHECKPOINT_STEPS = 2
# KEEP_CHECKPOINT_MAX = 5
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.piecewise_constant
step_per_epoch = 50000 // 200
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.01, 0.001, 0.0001, 0.00001],
    "boundaries": [step_per_epoch * 200, step_per_epoch * 300, step_per_epoch * 350],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.WEIGHT_DECAY_RATE = 0.0005

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    # Pad(2),
    Crop(size=IMAGE_SIZE),
    FlipLeftRight(),
])
