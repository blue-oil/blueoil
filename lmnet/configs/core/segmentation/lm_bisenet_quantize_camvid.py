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
from lmnet.networks.segmentation.lm_bisenet import LMBiSeNetQuantize
from lmnet.datasets.camvid import Camvid
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    PerImageStandardization,
)
from lmnet.post_processor import (
    Bilinear,
    Softmax,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = LMBiSeNetQuantize
DATASET_CLASS = Camvid

IMAGE_SIZE = [352, 480]
BATCH_SIZE = 8
DATA_FORMAT = "NHWC"
TASK = Tasks.SEMANTIC_SEGMENTATION
CLASSES = DATASET_CLASS.classes

MAX_EPOCHS = 400
SAVE_STEPS = 1000
TEST_STEPS = 1000
SUMMARISE_STEPS = 1000

# distributed training
IS_DISTRIBUTION = False

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# for debug
# BATCH_SIZE = 2
# SUMMARISE_STEPS = 1
# IS_DEBUG = True

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization(),
])
POST_PROCESSOR = Sequence([
    Bilinear(size=IMAGE_SIZE, data_format=DATA_FORMAT, compatible_tensorflow_v1=True),
    Softmax(),
])


NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.
NETWORK.AUXILIARY_LOSS_WEIGHT = 0.5
NETWORK.USE_FEATURE_FUSION = True
NETWORK.USE_ATTENTION_REFINEMENT = True
NETWORK.USE_TAIL_GAP = True
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Resize(size=IMAGE_SIZE),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    FlipLeftRight(),
    Hue((-10, 10)),
])
DATASET.ENABLE_PREFETCH = True
