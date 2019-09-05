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

from lmnet.common import Tasks
from lmnet.networks.optical_flow_estimation.flownet_s_v1 import FlowNetSV1
from lmnet.datasets.optical_flow_estimation import FlyingChairs
from lmnet.data_processor import Sequence
from lmnet.data_augmentor import (
    Brightness,
    Contrast,
    Color
)


IS_DEBUG = False

NETWORK_CLASS = FlowNetSV1
DATASET_CLASS = FlyingChairs

IMAGE_SIZE = [384, 512]
BATCH_SIZE = 8
DATA_FORMAT = "NHWC"
TASK = Tasks.OPTICAL_FLOW_ESTIMATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 1200000
SAVE_CHECKPOINT_STEPS = 50000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 10000
SUMMARISE_STEPS = 1000
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# distributed training
IS_DISTRIBUTION = False

# for debug
# MAX_STEPS = 10
# BATCH_SIZE = 31
# SAVE_CHECKPOINT_STEPS = 2
# TEST_STEPS = 10
# SUMMARISE_STEPS = 2
# IS_DEBUG = True

# TODO check
PRE_PROCESSOR = None
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9, "momentum2": 0.999}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    "boundaries": [400000, 600000, 800000, 1000000],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.VALIDATION_RATE = 0.2
DATASET.VALIDATION_SEED = 2019

# TODO I use default values because the metrics used in the paper are different.
# I didn't add Gaussian Blur because it's different from Gaussian Noise.
# Augmentation is not available in pytorch repo
DATASET.AUGMENTOR = Sequence([
    Brightness(),
    Contrast(),
    Color()
])
