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

from lmnet.common import Tasks
from lmnet.networks.instance_segmentation_old.resnet18_backbone import ResnetQuantize
#from lmnet.networks.classification.lm_resnet import LmResnetQuantize
from lmnet.datasets.ilsvrc_2012 import Ilsvrc2012
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    Crop,
    FlipLeftRight,
    Hue,
)

from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = ResnetQuantize
DATASET_CLASS = Ilsvrc2012

IMAGE_SIZE = [224, 224]
BATCH_SIZE = 4
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

# MAX_STEPS = 2000000
# SAVE_CHECKPOINT_STEPS = 50000
# KEEP_CHECKPOINT_MAX = 5
# TEST_STEPS = 50000
# SUMMARISE_STEPS = 50000
# USE_RECOVERY = True
# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""


# for debug
MAX_STEPS = 100
# BATCH_SIZE = 31
SAVE_CHECKPOINT_STEPS = 10
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 10
SUMMARISE_STEPS = 2
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
# NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
# NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
# NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
# NETWORK.LEARNING_RATE_KWARGS = {
#     "values": [0.1, 0.01, 0.001, 0.0001],
#     "boundaries": [40000, 60000, 80000],
# }
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
# NETWORK.WEIGHT_DECAY_RATE = 0.0001
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
    # Pad(2),
    Crop(size=IMAGE_SIZE),
    # FlipLeftRight(),
    # Crop(size=IMAGE_SIZE, resize=256),
    FlipLeftRight(),
    # Brightness((0.75, 1.25)),
    # Color((0.75, 1.25)),
    # Contrast((0.75, 1.25)),
    # Hue((-10, 10)),
])
DATASET.ENABLE_PREFETCH = True
