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

import numpy as np
import tensorflow as tf
from easydict import EasyDict

from lmnet.common import Tasks
from lmnet.data_processor import Sequence
from lmnet.networks.optical_flow_estimation.flownet_s_v3 import (
    FlowNetSV3
)
from lmnet.datasets.optical_flow_estimation import (
    FlyingChairs, ChairsSDHom
)
from lmnet.networks.optical_flow_estimation.data_augmentor import (
    Brightness, Color, Contrast, Gamma, GaussianBlur, GaussianNoise, Hue,
    FlipLeftRight, FlipTopBottom, Scale, Rotate, Translate
)
from lmnet.networks.optical_flow_estimation.pre_processor import (
    DevideBy255
)

NETWORK_CLASS = FlowNetSV3
DATASET_CLASS = FlyingChairs

IMAGE_SIZE = [384, 512]
DATA_FORMAT = "NHWC"
TASK = Tasks.OPTICAL_FLOW_ESTIMATION
CLASSES = DATASET_CLASS.classes

IS_DEBUG = False
MAX_STEPS = 1200000
SAVE_CHECKPOINT_STEPS = 5000
KEEP_CHECKPOINT_MAX = 20
TEST_STEPS = 250
SUMMARISE_STEPS = 1000
BATCH_SIZE = 8

# for debugging
# IS_DEBUG = True
# MAX_STEPS = 10
# BATCH_SIZE = 31
# SAVE_CHECKPOINT_STEPS = 2
# KEEP_CHECKPOINT_MAX = 5
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# distributed training
IS_DISTRIBUTION = False

PRE_PROCESSOR = Sequence([
    DevideBy255(),
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"beta1": 0.9, "beta2": 0.999}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    "boundaries": [400000, 600000, 800000, 1000000],
}
NETWORK.WEIGHT_DECAY_RATE = 0.0004
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.TRAIN_ENABLE_PREFETCH = True
DATASET.TRAIN_PROCESS_NUM = 10
DATASET.TRAIN_QUEUE_SIZE = 1000
DATASET.VALIDATION_ENABLE_PREFETCH = False
DATASET.VALIDATION_PRE_LOAD = False
DATASET.VALIDATION_PROCESS_NUM = 1
DATASET.VALIDATION_QUEUE_SIZE = 500
DATASET.VALIDATION_RATE = 0.1
DATASET.VALIDATION_SEED = 2019

# TODO I use default values because the metrics used in the paper are different.
# I didn't add Gaussian Blur because it's different from Gaussian Noise.
# Augmentation is not available in pytorch repo

# NOTE (by KI-42) in the FlowNetS paper, the following augmentations were used.
# Geometric transformation
# translation       U([-20 %, +20 %])
# rotation          U([-17 deg, +17 deg])
# scaling           U([0.9, 2.0])
# Pixel-Wise transformation
# Gaussian noise    N(0, 1) * U([0.0, 0.04 * (255)])
# contrast          U([0.2, 1.4])
# color             U([0.5, 2.0])
# gamma             U([0.7, 1.5])
# brightness        1 + 0.2 * N(0, 1)

# NOTE (by KI-42) in this setup, I modified the augmentation setup described above a little bit.
# hue               U([-128 deg, 128 deg])
# brightness        U(0.6, 1.4)

DATASET.AUGMENTOR = Sequence([
    # Geometric transformation
    FlipLeftRight(0.5),
    FlipTopBottom(0.5),
    Translate(-0.2, 0.2),
    Rotate(-17, +17),
    Scale(1.0, 2.0),
    # Pixel-wise augmentation
    Brightness(0.6, 1.4),
    Contrast(0.2, 1.4),
    Color(0.5, 2.0),
    Gamma(0.7, 1.5),
    # Hue(-128.0, 128.0),
    GaussianNoise(10.0)
    # GaussianBlur(0.0, 2.0)
])
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
