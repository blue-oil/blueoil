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
from lmnet.networks.optical_flow_estimation.lm_flownet_v1 import (
    LmFlowNet, LmFlowNetQuantized
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
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

NETWORK_CLASS = LmFlowNet
CONV_DEPTH = 5
DATASET_CLASS = FlyingChairs
SLICE_STEP = 4
IMAGE_SIZE = [384 // SLICE_STEP, 512 // SLICE_STEP]

DATA_FORMAT = "NHWC"
TASK = Tasks.OPTICAL_FLOW_ESTIMATION
CLASSES = DATASET_CLASS.classes

IS_DEBUG = False
MAX_STEPS = 1200000
SAVE_CHECKPOINT_STEPS = 10000
KEEP_CHECKPOINT_MAX = 120
TEST_STEPS = 1000
SUMMARISE_STEPS = 1000
BATCH_SIZE = 8

IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

IS_DISTRIBUTION = False

AUGMENTOR = Sequence([
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
NETWORK.CONV_DEPTH = CONV_DEPTH
NETWORK.DIV_FLOW = 20.0
NETWORK.WEIGHT_DECAY_RATE = 0.0004
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.SLICE_STEP = SLICE_STEP
DATASET.TRAIN_ENABLE_PREFETCH = True
DATASET.TRAIN_PROCESS_NUM = 10
DATASET.TRAIN_QUEUE_SIZE = 1000
DATASET.VALIDATION_ENABLE_PREFETCH = False
DATASET.VALIDATION_PRE_LOAD = False
DATASET.VALIDATION_PROCESS_NUM = 1
DATASET.VALIDATION_QUEUE_SIZE = 500
DATASET.VALIDATION_RATE = 0.1
DATASET.VALIDATION_SEED = 2019

DATASET.AUGMENTOR = AUGMENTOR
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
