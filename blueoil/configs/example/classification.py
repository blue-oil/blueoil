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
from blueoil.networks.classification.lmnet_v0 import LmnetV0Quantize
from blueoil.datasets.cifar10 import Cifar10
from blueoil.data_processor import Sequence
from blueoil.pre_processor import (
    Resize,
    PerImageStandardization,
)
from blueoil.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
)
from blueoil.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = True

NETWORK_CLASS = LmnetV0Quantize
DATASET_CLASS = Cifar10

IMAGE_SIZE = [28, 28]
BATCH_SIZE = 32
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS.classes

KEEP_CHECKPOINT_MAX = 5
MAX_EPOCHS = 1  # MAX_STEPS = 1561
SAVE_CHECKPOINT_STEPS = 100
TEST_STEPS = 100
SUMMARISE_STEPS = 10


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = [
    "conv1/kernel:",
    "conv1/bias:",
    "conv2/kernel:",
    "conv2/bias:",
    "conv3/kernel:",
    "conv3/bias:",
    "conv4/kernel:",
    "conv4/bias:",
    "conv5/kernel:",
    "conv5/bias:",
    "conv6/kernel:",
    "conv6/bias:",
]
PRETRAIN_DIR = "saved/lmnet_0.01_caltech101/checkpoints"
PRETRAIN_FILE = "save.ckpt-99001"

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    PerImageStandardization()
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.compat.v1.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.WEIGHT_DECAY_RATE = 0.0005
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
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
])
