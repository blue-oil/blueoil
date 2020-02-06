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

from nn.common import Tasks
from nn.networks.keypoint_detection.{{network_module}} import {{network_class}}
from nn.datasets.{{dataset_module}} import {{dataset_class}}
from nn.data_processor import Sequence
from nn.pre_processor import (
    DivideBy255,
    ResizeWithJoints,
    JointsToGaussianHeatmap
)
from nn.post_processor import (
    GaussianHeatmapToJoints
)
from nn.data_augmentor import (
    Brightness,
    Color,
    Contrast
)
from nn.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = {{network_class}}
DATASET_CLASS = {{dataset_class}}

IMAGE_SIZE = {{image_size}}
BATCH_SIZE = {{batch_size}}
DATA_FORMAT = "NHWC"
TASK = Tasks.KEYPOINT_DETECTION
CLASSES = {{classes}}

MAX_EPOCHS = {{max_epochs}}
SAVE_CHECKPOINT_STEPS = {{save_checkpoint_steps}}
KEEP_CHECKPOINT_MAX = {{keep_checkpoint_max}}
TEST_STEPS = {{test_steps}}
SUMMARISE_STEPS = {{summarise_steps}}

# distributed training
IS_DISTRIBUTION = False

# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = []
PRETRAIN_DIR = ""
PRETRAIN_FILE = ""

# stride of output heatmap. the smaller, the slower.
STRIDE = 8

PRE_PROCESSOR = Sequence([
    ResizeWithJoints(image_size=IMAGE_SIZE),
    JointsToGaussianHeatmap(image_size=IMAGE_SIZE,
                            stride=STRIDE, sigma=2),
    DivideBy255()
])
POST_PROCESSOR = Sequence([
    GaussianHeatmapToJoints(num_dimensions=2, stride=STRIDE, confidence_threshold=0.1)
])

step_per_epoch = int(149813 / BATCH_SIZE)

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = {{optimizer_class}}
NETWORK.OPTIMIZER_KWARGS = {{optimizer_kwargs}}
NETWORK.LEARNING_RATE_FUNC = {{learning_rate_func}}
NETWORK.LEARNING_RATE_KWARGS = {{learning_rate_kwargs}}

NETWORK.STRIDE = STRIDE
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}

DATASET = EasyDict()
DATASET.IMAGE_SIZE = IMAGE_SIZE
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25))
])
DATASET.ENABLE_PREFETCH = True
