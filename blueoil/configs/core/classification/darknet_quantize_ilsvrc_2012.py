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
from blueoil.datasets.ilsvrc_2012 import Ilsvrc2012
from blueoil.data_processor import Sequence
from blueoil.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    Crop,
    FlipLeftRight,
    Hue,
)
from blueoil.networks.classification.darknet import DarknetQuantize
from blueoil.pre_processor import (
    Resize,
    DivideBy255,
)
from blueoil.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = DarknetQuantize
DATASET_CLASS = Ilsvrc2012

IMAGE_SIZE = [224, 224]
BATCH_SIZE = 48
DATA_FORMAT = "NCHW"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

MAX_STEPS = 2000000
SAVE_CHECKPOINT_STEPS = 50000
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 50000
SUMMARISE_STEPS = 10000

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
    DivideBy255()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.compat.v1.train.polynomial_decay
# TODO(wakiska): It is same as original yolov2 paper (batch size = 128).
NETWORK.LEARNING_RATE_KWARGS = {"learning_rate": 1e-1, "decay_steps": 1600000, "power": 4.0, "end_learning_rate": 0.0}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0,
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = True
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Crop(size=IMAGE_SIZE, resize=256),
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
])
