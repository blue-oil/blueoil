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
from lmnet.datasets.image_folder import ImageFolderBase
from lmnet.networks.classification.lmnet_v0 import LmnetV0Quantize
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    Resize,
    DivideBy255,
)
from lmnet.data_augmentor import (
    FlipLeftRight,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)


class Dummy(ImageFolderBase):
    extend_dir = "dummy_classification"


IS_DEBUG = False

NETWORK_CLASS = LmnetV0Quantize
DATASET_CLASS = Dummy


IMAGE_SIZE = [28, 28]
BATCH_SIZE = 2
DATA_FORMAT = "NHWC"
TASK = Tasks.CLASSIFICATION
CLASSES = DATASET_CLASS(subset="train", batch_size=1).classes

MAX_EPOCHS = 1
SAVE_CHECKPOINT_STEPS = 1
KEEP_CHECKPOINT_MAX = 5
TEST_STEPS = 100
SUMMARISE_STEPS = 100


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

PRE_PROCESSOR = Sequence([
    Resize(size=IMAGE_SIZE),
    DivideBy255(),
])
POST_PROCESSOR = None

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.AdamOptimizer
NETWORK.OPTIMIZER_KWARGS = {"learning_rate": 0.001}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
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
])
